#!/bin/bash
# -----------------------------------------------
# yulong 在新电脑上的 SATA 启动脚本 (已验证)
# -----------------------------------------------
# tmux new -s sata_train
# source start_sata.sh
# wandb login eb43c996f452e659b0970120ca7786927be14814
echo "正在激活 SATA 环境..."

# 1. 激活 Conda 环境 (来自 conda env list)
#    你的环境名称是 'sata-env'
conda activate sata-env

# 2. 添加 Conda 环境库路径 (来自 conda env list)
#    路径是 '/home/yulong/miniconda3/envs/sata-env'
export LD_LIBRARY_PATH=/home/yulong/miniconda3/envs/sata-env/lib:$LD_LIBRARY_PATH

# 3. 添加 Isaac Gym 的特定路径 (来自 find ... plugInfo.json)
#    (已从你的长列表中选出正确的 Isaac Gym SDK 路径)
export GYM_USD_PLUG_INFO_PATH=/home/yulong/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugInfo.json

echo "-------------------------------------"
echo "sata-env 环境激活完毕。"
echo "-------------------------------------"