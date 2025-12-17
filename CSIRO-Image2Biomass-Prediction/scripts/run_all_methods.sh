#!/bin/bash
# 一键运行四种方法并融合

echo "=================================================================="
echo "🚀 运行四维对齐框架的所有方法"
echo "=================================================================="

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BASE_DIR"

echo "项目目录: $BASE_DIR"

# 选择运行模式
echo ""
echo "【运行模式选择】"
echo "  1. 串行运行（稳定，约44分钟）"
echo "  2. 并行运行（快速，约15分钟，需要4核CPU）"
echo "  3. 只运行C2BA（最快，7分钟）"
echo ""
read -p "请选择模式 [1/2/3，默认3]: " MODE
MODE=${MODE:-3}

echo ""
echo "=================================================================="

if [ "$MODE" == "3" ]; then
    echo "【只运行C2BA快速版】"
    echo "=================================================================="
    
    echo ""
    echo "▶ 运行C2BA..."
    python methods/c2ba/c2ba_fast.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ C2BA完成"
        echo "   结果: results/c2ba/submission_c2ba_fast.csv"
    else
        echo "❌ C2BA失败"
        exit 1
    fi
    
elif [ "$MODE" == "2" ]; then
    echo "【并行运行四个方法】"
    echo "=================================================================="
    
    # 创建日志目录
    mkdir -p methods/c2ba/logs methods/s2ba/logs methods/d2ba/logs methods/g2ba/logs
    
    echo ""
    echo "▶ 启动C2BA（后台）..."
    python methods/c2ba/c2ba_fast.py > methods/c2ba/logs/c2ba_fast.log 2>&1 &
    PID_C2BA=$!
    
    echo "▶ 启动S2BA（后台）..."
    python methods/s2ba/s2ba_fast.py > methods/s2ba/logs/s2ba_fast.log 2>&1 &
    PID_S2BA=$!
    
    echo "▶ 启动D2BA（后台）..."
    python methods/d2ba/d2ba_fast.py > methods/d2ba/logs/d2ba_fast.log 2>&1 &
    PID_D2BA=$!
    
    echo "▶ 启动G2BA（后台）..."
    python methods/g2ba/g2ba_fast.py > methods/g2ba/logs/g2ba_fast.log 2>&1 &
    PID_G2BA=$!
    
    echo ""
    echo "等待所有方法完成..."
    echo "日志位置:"
    echo "  - methods/c2ba/logs/c2ba_fast.log"
    echo "  - methods/s2ba/logs/s2ba_fast.log"
    echo "  - methods/d2ba/logs/d2ba_fast.log"
    echo "  - methods/g2ba/logs/g2ba_fast.log"
    echo ""
    echo "（可运行 'tail -f methods/*/logs/*.log' 查看进度）"
    
    wait $PID_C2BA
    echo "✅ C2BA完成"
    
    wait $PID_S2BA
    echo "✅ S2BA完成"
    
    wait $PID_D2BA
    echo "✅ D2BA完成"
    
    wait $PID_G2BA
    echo "✅ G2BA完成"
    
    echo ""
    echo "所有方法完成！现在融合..."
    python ensemble_fusion.py
    
else
    echo "【串行运行四个方法】"
    echo "=================================================================="
    
    START_TIME=$(date +%s)
    
    echo ""
    echo "▶ [1/4] 运行C2BA..."
    python methods/c2ba/c2ba_fast.py
    if [ $? -eq 0 ]; then
        echo "   ✅ C2BA完成"
    else
        echo "   ❌ C2BA失败，停止运行"
        exit 1
    fi
    
    echo ""
    echo "▶ [2/4] 运行S2BA..."
    python methods/s2ba/s2ba_fast.py
    if [ $? -eq 0 ]; then
        echo "   ✅ S2BA完成"
    else
        echo "   ⚠️ S2BA失败，继续运行"
    fi
    
    echo ""
    echo "▶ [3/4] 运行D2BA..."
    python methods/d2ba/d2ba_fast.py
    if [ $? -eq 0 ]; then
        echo "   ✅ D2BA完成"
    else
        echo "   ⚠️ D2BA失败，继续运行"
    fi
    
    echo ""
    echo "▶ [4/4] 运行G2BA..."
    python methods/g2ba/g2ba_fast.py
    if [ $? -eq 0 ]; then
        echo "   ✅ G2BA完成"
    else
        echo "   ⚠️ G2BA失败，继续运行"
    fi
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo ""
    echo "=================================================================="
    echo "所有方法完成！总耗时: $((ELAPSED/60))分钟"
    echo "=================================================================="
    
    echo ""
    echo "▶ 融合预测..."
    python ensemble_fusion.py
fi

echo ""
echo "=================================================================="
echo "🎉 完成！"
echo "=================================================================="
echo ""
echo "【生成的文件】"
ls -lh results/*/submission*.csv 2>/dev/null | awk '{print "  "$9" ("$5")"}'
echo ""
echo "【推荐提交】"
echo "  单方法: results/c2ba/submission_c2ba_advanced.csv (如已运行)"
echo "  融合:   results/submission_ensemble_simple.csv"
echo ""

