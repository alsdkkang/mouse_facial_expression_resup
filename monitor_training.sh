#!/bin/bash
# Training Monitor Script
# Run this to check training progress

PROJECT_DIR="/Users/minakang/Desktop/mouse-facial-expressions-2023-main"
cd "$PROJECT_DIR"

echo "========================================="
echo "Training Progress Monitor"
echo "========================================="
echo ""

# Check if training is running
TRAIN_PID=$(ps aux | grep "train_task1_baseline_model.py" | grep -v grep | awk '{print $2}')

if [ -z "$TRAIN_PID" ]; then
    echo "❌ Training is NOT running"
    echo ""
    echo "To start training:"
    echo "  cd $PROJECT_DIR"
    echo "  python -u mouse_facial_expressions/models/train_task1_baseline_model.py --epochs 10 --dataset_version \"1.1\" 2>&1 | tee training_log.txt &"
    exit 1
else
    echo "✅ Training is RUNNING (PID: $TRAIN_PID)"
    
    # Get runtime
    RUNTIME=$(ps -p $TRAIN_PID -o etime= | tr -d ' ')
    echo "   Runtime: $RUNTIME"
    
    # Get CPU usage
    CPU=$(ps -p $TRAIN_PID -o %cpu= | tr -d ' ')
    echo "   CPU: ${CPU}%"
    
    # Get memory usage
    MEM=$(ps -p $TRAIN_PID -o %mem= | tr -d ' ')
    echo "   Memory: ${MEM}%"
fi

echo ""
echo "========================================="
echo "Checkpoints"
echo "========================================="

if [ -d "models/checkpoints" ] && [ "$(ls -A models/checkpoints 2>/dev/null)" ]; then
    echo "✅ Checkpoints found:"
    ls -lht models/checkpoints/ | head -10
else
    echo "⏳ No checkpoints yet (will appear after first epoch)"
fi

echo ""
echo "========================================="
echo "Training Log"
echo "========================================="

if [ -f "training_log.txt" ] && [ -s "training_log.txt" ]; then
    echo "✅ Log file has content:"
    echo ""
    tail -30 training_log.txt
else
    echo "⏳ Log file is empty (output is buffered)"
    echo "   - First output will appear when epoch starts"
    echo "   - Check back in 10-15 minutes"
fi

echo ""
echo "========================================="
echo "Quick Commands"
echo "========================================="
echo "Watch log in real-time:"
echo "  tail -f training_log.txt"
echo ""
echo "Check progress again:"
echo "  ./monitor_training.sh"
echo ""
echo "Stop training:"
echo "  kill $TRAIN_PID"
echo ""
