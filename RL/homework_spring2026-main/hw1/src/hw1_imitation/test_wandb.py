import wandb
import numpy as np

# 1️⃣ 初始化（必须在最前面）
run = wandb.init(
    project="test-online",
    name="video-test"
)

# 2️⃣ log 标量
wandb.log({"accuracy": 0.99})

# 3️⃣ log 视频
video = np.random.randint(0, 255, (30, 3, 64, 64), dtype=np.uint8)

wandb.log({
    "demo_video": wandb.Video(video, fps=10)
})

# 4️⃣ 结束（可选但推荐）
run.finish()