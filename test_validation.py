"""
语法和导入检查测试
验证所有Python文件的语法正确性
"""

import ast
import os
import sys
from pathlib import Path


def check_python_syntax(file_path):
    """检查Python文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def find_python_files(root_dir):
    """查找所有Python文件"""
    python_files = []
    
    for path in Path(root_dir).rglob("*.py"):
        # 跳过虚拟环境和缓存目录
        if any(part in path.parts for part in ['.venv', 'venv', '__pycache__', '.git']):
            continue
        python_files.append(path)
    
    return sorted(python_files)


def check_project_structure():
    """检查项目结构"""
    print("Checking project structure...")
    
    required_dirs = [
        'src',
        'src/models',
        'src/data',
        'src/utils',
        'experiments',
        'experiments/config',
        'docs',
    ]
    
    required_files = [
        'README.md',
        'requirements.txt',
        'src/train_baseline.py',
        'src/train_compressed.py',
        'src/evaluate.py',
        'experiments/run_ablation.sh',
    ]
    
    all_ok = True
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"  ✓ Directory exists: {dir_path}")
        else:
            print(f"  ✗ Missing directory: {dir_path}")
            all_ok = False
    
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"  ✓ File exists: {file_path}")
        else:
            print(f"  ✗ Missing file: {file_path}")
            all_ok = False
    
    return all_ok


def run_tests():
    """运行所有测试"""
    print("="*60)
    print("Project Validation Tests")
    print("="*60)
    
    # 检查项目结构
    print("\n1. Project Structure Check")
    print("-"*60)
    structure_ok = check_project_structure()
    
    # 检查Python语法
    print("\n2. Python Syntax Check")
    print("-"*60)
    
    python_files = find_python_files('.')
    print(f"Found {len(python_files)} Python files")
    
    syntax_errors = []
    for file_path in python_files:
        success, error = check_python_syntax(file_path)
        if success:
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path}: {error}")
            syntax_errors.append((file_path, error))
    
    # 检查关键模块
    print("\n3. Key Module Check")
    print("-"*60)
    
    key_modules = [
        'src/models/compressed_bert.py',
        'src/models/distillation.py',
        'src/models/pruning.py',
        'src/models/quantization.py',
        'src/data/dataset_loader.py',
        'src/utils/metrics.py',
        'src/utils/visualization.py',
    ]
    
    modules_ok = True
    for module in key_modules:
        if os.path.isfile(module):
            success, error = check_python_syntax(module)
            if success:
                # 检查文件大小（确保不是空文件）
                size = os.path.getsize(module)
                if size > 100:
                    print(f"  ✓ {module} ({size} bytes)")
                else:
                    print(f"  ⚠ {module} seems too small ({size} bytes)")
                    modules_ok = False
            else:
                print(f"  ✗ {module}: {error}")
                modules_ok = False
        else:
            print(f"  ✗ Missing: {module}")
            modules_ok = False
    
    # 检查配置文件
    print("\n4. Configuration Files Check")
    print("-"*60)
    
    config_files = [
        'experiments/config/baseline.yaml',
        'experiments/config/distillation.yaml',
        'experiments/config/pruning.yaml',
        'experiments/config/quantization.yaml',
        'experiments/config/hybrid.yaml',
    ]
    
    configs_ok = True
    for config in config_files:
        if os.path.isfile(config):
            size = os.path.getsize(config)
            print(f"  ✓ {config} ({size} bytes)")
        else:
            print(f"  ✗ Missing: {config}")
            configs_ok = False
    
    # 检查文档
    print("\n5. Documentation Check")
    print("-"*60)
    
    doc_files = [
        'README.md',
        'docs/experiment_design.md',
        'docs/technical_details.md',
    ]
    
    docs_ok = True
    for doc in doc_files:
        if os.path.isfile(doc):
            size = os.path.getsize(doc)
            # 文档应该有足够的内容
            if size > 500:
                print(f"  ✓ {doc} ({size} bytes)")
            else:
                print(f"  ⚠ {doc} seems incomplete ({size} bytes)")
                docs_ok = False
        else:
            print(f"  ✗ Missing: {doc}")
            docs_ok = False
    
    # 总结
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_tests = [
        ("Project Structure", structure_ok),
        ("Python Syntax", len(syntax_errors) == 0),
        ("Key Modules", modules_ok),
        ("Configuration Files", configs_ok),
        ("Documentation", docs_ok),
    ]
    
    for test_name, success in all_tests:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, success in all_tests if success)
    total = len(all_tests)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if syntax_errors:
        print(f"\nSyntax Errors Found: {len(syntax_errors)}")
        for file_path, error in syntax_errors:
            print(f"  - {file_path}: {error}")
    
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    
    if success:
        print("\n✓ All validation tests passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run quick start: python quick_start.py")
        print("3. Train baseline: python src/train_baseline.py")
        print("4. Run ablation: bash experiments/run_ablation.sh")
    else:
        print("\n✗ Some tests failed. Please review the output above.")
    
    sys.exit(0 if success else 1)
