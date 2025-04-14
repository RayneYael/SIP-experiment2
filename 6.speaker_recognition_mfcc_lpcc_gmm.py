import json
import os
import numpy as np
import librosa
# from python_speech_features.base import lpcc # 不再需要这个库的 LPCC
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler # 可选，用于特征标准化
import warnings
from tqdm import tqdm # 用于显示进度条

# --- 配置参数 ---
# JSON 文件路径 (请根据你的实际路径修改)
train_json_path = r".\trainset.json"
test_json_path = r".\testset.json"
# 数据集根目录 (json文件中audio_filepath的相对起点, 请根据你的实际路径修改)
base_data_dir = r"./"

# MFCC 特征参数
N_MFCC = 13         # MFCC 阶数
SAMPLE_RATE = 16000 # TIMIT 采样率通常是 16kHz
N_FFT = 512         # FFT 点数 (通常是2的幂，对应约30ms @16kHz)
HOP_LENGTH = 160    # 帧移 (对应 10ms @16kHz)
WIN_LENGTH = 400    # 窗长 (对应 25ms @16kHz)

# LPCC 特征参数 (使用你的函数)
LPCC_ORDER = 13     # LPCC 阶数 (与 N_MFCC 保持一致，便于观察)
PRE_EMPHASIS = 0.97 # 预加重系数 (来自你的函数)
# 帧参数需要转换成秒，以匹配你的函数接口
LPCC_WIN_LENGTH_SEC = WIN_LENGTH / SAMPLE_RATE # 窗长 (秒)
LPCC_HOP_LENGTH_SEC = HOP_LENGTH / SAMPLE_RATE # 帧移 (秒)

# GMM 模型参数
N_COMPONENTS = 32   # GMM 组件数 (可调超参数)
COVARIANCE_TYPE = 'diag' # 协方差类型 ('diag', 'full', 'tied', 'spherical')
MAX_ITER = 100      # GMM 训练最大迭代次数
N_INIT = 3          # GMM 训练初始化次数 (增加稳定性)

# 是否使用特征标准化 (可选但推荐)
USE_SCALER = True

# --- 辅助函数 ---

def load_dataset(json_path):
    """从 JSON 文件加载数据集信息"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载数据集信息: {json_path}")
        return data
    except FileNotFoundError:
        print(f"错误: JSON 文件未找到: {json_path}")
        return None
    except Exception as e:
        print(f"加载 JSON 文件时出错: {e}")
        return None

# --- 使用你提供的 LPCC 函数 (稍作修改以强制采样率) ---
def extract_lpcc_features(audio_path, target_sr, n_lpcc=13, pre_emphasis=0.97, n_fft=512, win_length=0.025, hop_length=0.01):
    """
    从音频文件中提取LPCC特征 (修改版：强制使用 target_sr)

    参数:
    audio_path: 音频文件路径
    target_sr: 目标采样率 (例如 16000)
    n_lpcc: LPCC系数数量
    pre_emphasis: 预加重因子
    n_fft: FFT大小 (注意：在此实现中未直接用于LPC，但保留以维持接口)
    win_length: 窗口长度(秒)
    hop_length: 窗口移动步长(秒)

    返回:
    lpcc: LPCC特征矩阵 (时间帧数 × LPCC系数数) 或 None (如果出错)
    """
    try:
        # 加载音频，强制使用目标采样率
        y, sr = librosa.load(audio_path, sr=target_sr) # *** 关键修改 ***

        # 检查音频长度是否足够分帧
        win_length_samples = int(win_length * sr)
        if len(y) < win_length_samples:
             warnings.warn(f"音频 {os.path.basename(audio_path)} 太短 ({len(y)} samples)，无法提取LPCC，跳过。")
             return None

        # 预加重处理
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

        # 转换窗口长度和步长从秒到样本数
        hop_length_samples = int(hop_length * sr)

        # 分帧
        # 使用 librosa.stft 的底层帧逻辑，它处理边界情况更鲁棒
        # 或者保持 librosa.util.frame，但要注意它可能不包含末尾不足一帧的数据
        # 为了与 librosa.feature.mfcc 的帧数更可能一致，使用与 MFCC 相同的参数计算帧
        # frames = librosa.util.frame(y, frame_length=win_length_samples, hop_length=hop_length_samples)
        # 使用 STFT 获取加窗帧可能更标准，尽管我们只需要幅度
        # 或者直接用原始的分帧逻辑：
        frames = librosa.util.frame(y, frame_length=win_length_samples, hop_length=hop_length_samples).T
        # shape (n_frames, win_length_samples)

        if frames.shape[0] == 0:
             warnings.warn(f"分帧后帧数为 0: {os.path.basename(audio_path)}，跳过 LPCC 提取。")
             return None

        # 应用汉明窗
        window = np.hamming(win_length_samples)
        windowed_frames = frames * window

        # 逐帧提取LPC系数
        # librosa.lpc 的 order 参数是 LPC 模型的阶数 p
        # 它返回的 a 数组长度是 p+1, a[0] 是预测误差能量/增益, a[1:] 是 -(滤波器系数)
        lpc_coeffs_list = []
        for frame in windowed_frames:
            # order 参数应该是 LPCC_ORDER (即希望的滤波器阶数)
            a = librosa.lpc(frame, order=n_lpcc)
            lpc_coeffs_list.append(a)

        lpc_coeffs_arr = np.array(lpc_coeffs_list) # shape (n_frames, n_lpcc + 1)

        # 将LPC系数转换为LPCC系数
        lpcc_features = []
        for i in range(lpc_coeffs_arr.shape[0]):
            a = lpc_coeffs_arr[i] # a = [gain, -a1, -a2, ...]
            error_gain = a[0]
            lpc_filter_coeffs = -a[1:] # lpc_filter_coeffs = [a1, a2, ...]

            if error_gain <= 1e-12: # 避免 log(0) 或负数
                # print(f"Warning: Near-zero error gain encountered in frame {i} of {os.path.basename(audio_path)}")
                # 对于零增益帧，可以返回零 LPCC 或跳过该帧，或用小值代替
                # 这里我们用一个非常小的 gain 来计算，或者直接用全零？用全零可能更安全
                # lpcc = np.zeros(n_lpcc)
                # 或者尝试用一个小值计算
                 error_gain = 1e-12

            # 重新实现一个基于标准公式的转换可能更可靠
            # c[0] = ln(gain)
            # c[m] = -a[m] - sum_{k=1}^{m-1} ( (m-k)/m ) * c[m-k] * a[k]  for 1 <= m <= p
            # 使用你原来的转换逻辑:
            lpcc_frame = [0.0] * n_lpcc # 使用浮点数

            # 第一个LPCC系数 - 对数增益是常见的做法
            lpcc_frame[0] = np.log(error_gain) # 你原来的代码是 -np.log(a[0])? 确认下符号和含义

            # 剩余的LPCC系数 (使用你提供的公式逻辑)
            # 注意: a 在你的代码里是指 lpc_coeffs[i] (librosa 输出)
            # 你的公式中 a[n] 对应 librosa 输出的 a[n] (即 -真实滤波器系数 a_n)
            # 你的公式中 a[k] 对应 librosa 输出的 a[k]
            # 你的 lpcc[n-k] 是指已计算的 LPCC 系数

            for n in range(1, n_lpcc): # 计算 c_1 到 c_{n_lpcc-1}
                 # 确保索引在 lpc_filter_coeffs (即 a[1:]) 范围内
                 if n < len(a): # n 对应 librosa a 的索引
                     lpcc_val = a[n] #  这是 librosa 的 a[n] = -真实a_n
                     for k in range(1, n): # 这里的 k 是中间求和变量
                         # 你的公式: += a[k] * lpcc[n-k] * (n-k) / n
                         # 需要确认: lpcc[n-k] 指的是 lpcc_frame[n-k] (下标从0开始)
                         # a[k] 指的是 librosa a[k]
                         if (n - k) < len(lpcc_frame) and k < len(a):
                              lpcc_val += a[k] * lpcc_frame[n-k] * (n-k) / n
                     lpcc_frame[n] = -lpcc_val # 最终取负
                 else:
                      # 如果 n 超出了 librosa lpc 输出的长度 (应该不会发生，因为 order=n_lpcc)
                      lpcc_frame[n] = 0.0 # 或其他处理

            lpcc_features.append(lpcc_frame)

        return np.array(lpcc_features)

    except Exception as e:
        warnings.warn(f"处理文件 {audio_path} 提取 LPCC 时发生错误: {e}")
        import traceback
        traceback.print_exc() # 打印详细错误跟踪
        return None


def extract_fused_features(audio_path, sr):
    """提取 MFCC 和 LPCC 特征并融合"""
    try:
        # 1. 加载音频 (MFCC 部分会自己加载，但 LPCC 函数也需要路径)
        # 检查文件存在性
        if not os.path.exists(audio_path):
             warnings.warn(f"音频文件未找到: {audio_path}")
             return None

        # 2. 提取 MFCC 特征
        # librosa.load 在 mfcc 内部会被调用
        y_for_mfcc, sr_loaded = librosa.load(audio_path, sr=sr)
        if sr_loaded != sr:
            warnings.warn(f"音频 {os.path.basename(audio_path)} 采样率 ({sr_loaded}) 与目标 ({sr}) 不符，已重采样.")

        # 检查音频长度是否足够进行分帧
        if len(y_for_mfcc) < WIN_LENGTH:
            warnings.warn(f"音频 {os.path.basename(audio_path)} 太短 ({len(y_for_mfcc)} samples)，无法提取MFCC，跳过。")
            return None

        mfcc_features = librosa.feature.mfcc(y=y_for_mfcc, sr=sr, n_mfcc=N_MFCC,
                                             n_fft=N_FFT, hop_length=HOP_LENGTH,
                                             win_length=WIN_LENGTH)
        mfcc_features = mfcc_features.T # 转置为 (n_frames, n_mfcc)

        if mfcc_features.shape[0] == 0:
            warnings.warn(f"提取 MFCC 后帧数为 0: {os.path.basename(audio_path)}，跳过。")
            return None

        # 3. 提取 LPCC 特征 (使用你的函数)
        lpcc_features = extract_lpcc_features(
            audio_path=audio_path,
            target_sr=sr,             # 使用统一的采样率
            n_lpcc=LPCC_ORDER,
            pre_emphasis=PRE_EMPHASIS,
            n_fft=N_FFT,              # 传递 n_fft
            win_length=LPCC_WIN_LENGTH_SEC, # 传递秒为单位的窗长
            hop_length=LPCC_HOP_LENGTH_SEC  # 传递秒为单位的帧移
        )

        if lpcc_features is None or lpcc_features.shape[0] == 0:
            warnings.warn(f"未能为 {os.path.basename(audio_path)} 提取有效的 LPCC 特征。")
            return None # 如果 LPCC 提取失败，则无法融合

        # 4. 对齐帧数 (非常重要!)
        min_frames = min(mfcc_features.shape[0], lpcc_features.shape[0])

        if min_frames == 0:
             warnings.warn(f"MFCC 或 LPCC 提取后帧数为 0: {os.path.basename(audio_path)}，跳过。")
             return None

        mfcc_features = mfcc_features[:min_frames, :]
        lpcc_features = lpcc_features[:min_frames, :]

        # 5. 融合特征 (水平拼接)
        fused_features = np.hstack((mfcc_features, lpcc_features))
        # print(f"Debug: {os.path.basename(audio_path)} MFCC shape: {mfcc_features.shape}, LPCC shape: {lpcc_features.shape}, Fused shape: {fused_features.shape}")

        return fused_features

    except Exception as e:
        warnings.warn(f"处理文件 {audio_path} 提取融合特征时发生错误: {e}")
        import traceback
        traceback.print_exc() # 打印详细错误跟踪
        return None

# --- 主逻辑 (保持不变) ---

# 1. 加载训练和测试数据信息
print("--- 开始加载数据集信息 ---")
train_data = load_dataset(train_json_path)
test_data = load_dataset(test_json_path)

if not train_data or not test_data:
    print("错误：无法加载必要的 JSON 文件，程序终止。")
    exit()

# 2. 训练阶段：为每个说话人提取特征并训练 GMM
print("\n--- 开始训练阶段 ---")
speaker_models = {} # 存储 speaker_id -> GMM 模型的映射
speaker_features_temp = {} # 临时存储 speaker_id -> [特征列表] 的映射 (改名以示区分)
scalers = {} # 存储 speaker_id -> StandardScaler 对象的映射 (如果 USE_SCALER)

print("步骤 1/2: 提取训练特征...")
for item in tqdm(train_data, desc="提取训练特征"):
    speaker_id = item['speaker_id']
    relative_audio_path = item['audio_filepath']
    # 构建绝对路径
    audio_path = os.path.normpath(os.path.join(base_data_dir, relative_audio_path))

    # 注意：这里 extract_fused_features 内部会检查文件存在性
    features = extract_fused_features(audio_path, SAMPLE_RATE)

    if features is not None and features.shape[0] > 0: # 确保有有效的特征帧
        if speaker_id not in speaker_features_temp:
            speaker_features_temp[speaker_id] = []
        speaker_features_temp[speaker_id].append(features)
    # else: # 可以在这里加日志记录哪些文件提取失败
        # print(f"Info: Skipped {os.path.basename(audio_path)} due to feature extraction issues.")


print(f"\n收集到 {len(speaker_features_temp)} 个说话人的特征数据。")
print("\n步骤 2/2: 训练 GMM 模型...")
trained_speakers = list(speaker_features_temp.keys())
for speaker_id in tqdm(trained_speakers, desc="训练 GMM"):
    all_features_list = speaker_features_temp[speaker_id]
    if not all_features_list:
        # This case should theoretically not happen if appending logic is correct
        warnings.warn(f"说话人 {speaker_id} 没有有效的训练特征列表，跳过。")
        continue

    # 将该说话人的所有特征帧合并成一个大数组
    training_features = np.vstack(all_features_list)

    # 检查特征数量是否足够训练 GMM
    min_samples_required = N_COMPONENTS * 5 # 启发式规则
    if training_features.shape[0] < min_samples_required or training_features.shape[0] < N_COMPONENTS:
        warnings.warn(f"说话人 {speaker_id} 只有 {training_features.shape[0]} 个特征帧，"
                      f"可能不足以训练 {N_COMPONENTS} 个组件的 GMM，跳过该说话人。")
        continue

    try:
        # (可选) 特征标准化
        current_features = training_features # 避免修改原始 training_features
        if USE_SCALER:
            scaler = StandardScaler()
            current_features = scaler.fit_transform(training_features)
            scalers[speaker_id] = scaler # 存储 scaler 以便测试时使用

        # 训练 GMM
        gmm = GaussianMixture(n_components=N_COMPONENTS, covariance_type=COVARIANCE_TYPE,
                              max_iter=MAX_ITER, n_init=N_INIT, random_state=42,
                              verbose=0, reg_covar=1e-6)

        gmm.fit(current_features) # 使用可能被标准化的特征进行训练

        if gmm.converged_:
            speaker_models[speaker_id] = gmm
        else:
            warnings.warn(f"说话人 {speaker_id} 的 GMM 训练未收敛。")

    except ValueError as ve:
         warnings.warn(f"训练说话人 {speaker_id} 的 GMM 时出现 ValueError: {ve}. 可能数据不足或特征值有问题。")
    except Exception as e:
        warnings.warn(f"训练说话人 {speaker_id} 的 GMM 时发生意外错误: {e}")
        import traceback
        traceback.print_exc()

# 清理不再需要的原始特征数据以释放内存
del speaker_features_temp

print(f"\n--- 训练完成 ---")
print(f"成功训练了 {len(speaker_models)} 个说话人的 GMM 模型。")
if not speaker_models:
    print("错误：没有成功训练任何说话人模型，无法进行测试。程序终止。")
    exit()

# 3. 测试阶段：对测试集进行识别并评估准确率
print("\n--- 开始测试阶段 ---")
correct_predictions = 0
total_predictions = 0
unknown_speakers_in_test = 0 # 记录测试集中出现但训练集中没有的说话人
prediction_failures = 0 # 记录无法进行预测的文件数

available_model_speakers = set(speaker_models.keys())

for item in tqdm(test_data, desc="测试语音"):
    true_speaker_id = item['speaker_id']
    relative_audio_path = item['audio_filepath']
    audio_path = os.path.normpath(os.path.join(base_data_dir, relative_audio_path))

    # 检查该说话人是否在训练模型中存在 (可选评估策略)
    # if true_speaker_id not in available_model_speakers:
    #     unknown_speakers_in_test += 1
    #     continue # 跳过未知说话人的评估

    total_predictions += 1

    # 提取测试文件的融合特征
    test_features = extract_fused_features(audio_path, SAMPLE_RATE)

    if test_features is None or test_features.shape[0] == 0:
        warnings.warn(f"无法为测试文件 {os.path.basename(audio_path)} 提取有效特征，无法进行预测。")
        prediction_failures += 1
        continue # 跳过这个文件的预测

    best_score = -np.inf
    predicted_speaker_id = None

    # 计算该测试语音在每个说话人模型下的对数似然得分
    for speaker_id, gmm in speaker_models.items():
        try:
            # 如果使用了标准化，需要用对应说话人的 scaler 来转换测试特征
            current_test_features = test_features.copy() # 使用副本以防 transform 修改原数组
            if USE_SCALER and speaker_id in scalers:
                current_test_features = scalers[speaker_id].transform(current_test_features)
            elif USE_SCALER and speaker_id not in scalers:
                 # 这通常不应该发生，除非训练时跳过了这个 speaker_id
                 warnings.warn(f"测试时找不到说话人 {speaker_id} 的 scaler，将使用原始特征。")

            # GMM.score() 返回平均对数似然
            score = gmm.score(current_test_features)

            if score > best_score:
                best_score = score
                predicted_speaker_id = speaker_id

        except Exception as e:
            warnings.warn(f"计算说话人 {speaker_id} 对文件 {os.path.basename(audio_path)} 的得分时出错: {e}")
            # 可以在这里打印更详细的错误信息
            continue # 跳过这个模型的评分

    # 比较预测结果和真实标签
    if predicted_speaker_id is not None:
        # print(f"文件: {os.path.basename(audio_path)}, 真实: {true_speaker_id}, 预测: {predicted_speaker_id}, 最高分: {best_score:.2f}")
        if predicted_speaker_id == true_speaker_id:
            correct_predictions += 1
    else:
        warnings.warn(f"未能为文件 {os.path.basename(audio_path)} 产生预测结果（所有模型评分都失败或特征提取失败）。")
        # prediction_failures 计数已在特征提取失败时增加


# 4. 计算并输出准确率
print("\n--- 测试完成 ---")
valid_predictions = total_predictions - prediction_failures
if valid_predictions > 0:
    accuracy = (correct_predictions / valid_predictions) * 100
    print(f"\n识别准确率: {accuracy:.2f}% ({correct_predictions} / {valid_predictions})")
    print(f"(总测试样本: {total_predictions}, 预测失败: {prediction_failures})")

else:
    print(f"没有进行任何有效的预测（总测试样本: {total_predictions}, 预测失败: {prediction_failures}），无法计算准确率。")

if unknown_speakers_in_test > 0:
     print(f"注意：测试集中有 {unknown_speakers_in_test} 条记录来自训练集中未出现的说话人 (这些记录当前被跳过评估)。")

print("\n--- 程序结束 ---")