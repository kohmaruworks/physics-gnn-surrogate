import torch
import time

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    print("GPUで行列の掛け算を実行中...")
    start_time = time.time()
    
    # 10,000 x 10,000 の巨大な行列を2つ作成してGPUに転送
    x = torch.randn(10000, 10000).cuda()
    y = torch.randn(10000, 10000).cuda()
    
    z = torch.matmul(x, y)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"計算完了！ 所要時間: {end_time - start_time:.4f} 秒")
    print("RTX 5080は正常に稼働しています。")
else:
    print("CUDAが認識されていません。")