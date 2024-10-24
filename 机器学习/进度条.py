import time

# 输出训练进度
def print_progress(i, start, end, step):
    # 计算进度百分比
    progress_percentage = (i - start) / (end - start) * 100
    progress_bar = "▋" * int(progress_percentage // 2)  # 每 2% 显示一个进度块
    print("\rTraining progress: {:.0f}%: {}".format(progress_percentage, progress_bar), end="")
    time.sleep(0.05)

for i in range(0, 100 + 1, 10):
    print_progress(i, 0, 100, 10)
