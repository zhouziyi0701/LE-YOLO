import torch
import torch.nn as nn

 
try:
    from zerodce_module import ZeroDCEPlusPlusYOLO as OriginalZeroDCEModule

except ImportError:
    print("="*80)
    print("❌ 导入错误：无法在 'zerodce_module.py' 文件中找到 'ZeroDCEPlusPlusYOLO' 类。")
    print("请确保你已经将 'zerodce_module (1).py' 文件重命名为 'zerodce_module.py'")
    print("并且它和你运行的主脚本在同一个文件夹里。")
    print("="*80)
    raise


class ZeroDCEPlusPlusYOLO(nn.Module):
       
    def __init__(self, c1, c2, *args):
        super().__init__()
        
        print(f"✅ 成功加载自定义模块 'ZeroDCEPlusPlusYOLO' (包装器)。")
        print(f"  > 已接收并忽略来自 YAML 的参数: c1={c1}, c2={c2}, args={args}")
        
        self.real_model = OriginalZeroDCEModule() 
    
    def forward(self, x):
        return self.real_model(x)
