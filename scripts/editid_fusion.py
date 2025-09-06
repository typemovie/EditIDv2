#!/usr/bin/env python3
"""
交互式SafeTensors权重融合工具
先进行key对比分析，然后让用户选择是否融合以及融合方式
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Set
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from datetime import datetime


class InteractiveSafeTensorsMerger:
    """交互式SafeTensors文件融合器"""

    def __init__(self):
        self.file1_tensors = {}
        self.file2_tensors = {}
        self.file1_metadata = {}
        self.file2_metadata = {}
        self.file1_info = {}
        self.file2_info = {}
        self.file1_path = ""
        self.file2_path = ""

    def load_safetensors_info(self, file_path: str) -> Tuple[Dict[str, dict], Dict[str, str]]:
        """加载safetensors文件信息（只获取key和shape信息，不加载实际数据）"""
        tensor_info = {}
        metadata = {}

        try:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                # 获取元数据
                metadata = f.metadata() or {}

                # 获取所有张量的基本信息
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    tensor_info[key] = {
                        'shape': tensor.shape,
                        'dtype': tensor.dtype,
                        'size': tensor.numel()
                    }

            return tensor_info, metadata

        except Exception as e:
            print(f"✗ 加载失败: {file_path}")
            print(f"  错误: {str(e)}")
            raise

    def analyze_compatibility(self, file1_info: Dict[str, dict], file2_info: Dict[str, dict]) -> dict:
        """分析两个文件的兼容性"""
        keys1 = set(file1_info.keys())
        keys2 = set(file2_info.keys())

        common_keys = keys1.intersection(keys2)
        only_in_file1 = keys1 - keys2
        only_in_file2 = keys2 - keys1

        # 分析相同key的兼容性
        compatible_keys = set()
        shape_mismatch_keys = set()
        dtype_mismatch_keys = set()

        for key in common_keys:
            info1 = file1_info[key]
            info2 = file2_info[key]

            if info1['shape'] != info2['shape']:
                shape_mismatch_keys.add(key)
            elif info1['dtype'] != info2['dtype']:
                dtype_mismatch_keys.add(key)
            else:
                compatible_keys.add(key)

        return {
            'keys1': keys1,
            'keys2': keys2,
            'common_keys': common_keys,
            'only_in_file1': only_in_file1,
            'only_in_file2': only_in_file2,
            'compatible_keys': compatible_keys,
            'shape_mismatch_keys': shape_mismatch_keys,
            'dtype_mismatch_keys': dtype_mismatch_keys
        }

    def display_analysis_summary(self, analysis: dict) -> None:
        """显示分析摘要"""
        print(f"\n{'=' * 60}")
        print(f"📊 文件对比分析摘要")
        print(f"{'=' * 60}")

        print(f"📈 基本统计:")
        print(f"  文件1 key数量: {len(analysis['keys1'])}")
        print(f"  文件2 key数量: {len(analysis['keys2'])}")
        print(f"  相同 key数量: {len(analysis['common_keys'])}")
        print(f"  仅文件1独有: {len(analysis['only_in_file1'])}")
        print(f"  仅文件2独有: {len(analysis['only_in_file2'])}")

        print(f"\n🔍 兼容性分析:")
        print(f"  完全兼容: {len(analysis['compatible_keys'])} 个")
        print(f"  形状不匹配: {len(analysis['shape_mismatch_keys'])} 个")
        print(f"  类型不匹配: {len(analysis['dtype_mismatch_keys'])} 个")

        # 兼容性评估
        total_common = len(analysis['common_keys'])
        if total_common > 0:
            compatibility_rate = len(analysis['compatible_keys']) / total_common * 100
            print(f"  兼容率: {compatibility_rate:.1f}%")

            if compatibility_rate >= 95:
                print(f"  ✅ 高度兼容，推荐融合")
            elif compatibility_rate >= 80:
                print(f"  ⚠️  中等兼容，可谨慎融合")
            else:
                print(f"  ❌ 兼容性较低，不推荐融合")

        print(f"{'=' * 60}")

    def display_detailed_analysis(self, analysis: dict) -> None:
        """显示详细分析"""
        print(f"\n🔍 详细Key分析:")

        # 显示形状不匹配的key
        if analysis['shape_mismatch_keys']:
            print(f"\n⚠️  形状不匹配的Key ({len(analysis['shape_mismatch_keys'])}个):")
            for i, key in enumerate(sorted(analysis['shape_mismatch_keys'])):
                if i >= 5:  # 只显示前5个
                    print(f"    ... 还有 {len(analysis['shape_mismatch_keys']) - 5} 个")
                    break
                info1 = self.file1_info[key]
                info2 = self.file2_info[key]
                print(f"    {key}: {info1['shape']} vs {info2['shape']}")

        # 显示类型不匹配的key
        if analysis['dtype_mismatch_keys']:
            print(f"\n⚠️  类型不匹配的Key ({len(analysis['dtype_mismatch_keys'])}个):")
            for i, key in enumerate(sorted(analysis['dtype_mismatch_keys'])):
                if i >= 5:  # 只显示前5个
                    print(f"    ... 还有 {len(analysis['dtype_mismatch_keys']) - 5} 个")
                    break
                info1 = self.file1_info[key]
                info2 = self.file2_info[key]
                print(f"    {key}: {info1['dtype']} vs {info2['dtype']}")

        # 显示独有的key
        if analysis['only_in_file1']:
            print(f"\n📋 仅在文件1中的Key ({len(analysis['only_in_file1'])}个):")
            for i, key in enumerate(sorted(analysis['only_in_file1'])):
                if i >= 5:  # 只显示前5个
                    print(f"    ... 还有 {len(analysis['only_in_file1']) - 5} 个")
                    break
                print(f"    {key}")

        if analysis['only_in_file2']:
            print(f"\n📋 仅在文件2中的Key ({len(analysis['only_in_file2'])}个):")
            for i, key in enumerate(sorted(analysis['only_in_file2'])):
                if i >= 5:  # 只显示前5个
                    print(f"    ... 还有 {len(analysis['only_in_file2']) - 5} 个")
                    break
                print(f"    {key}")

    def get_user_choice_merge(self) -> bool:
        """询问用户是否要进行融合"""
        print(f"\n{'=' * 60}")
        print("🤔 是否要进行权重融合？")
        print("1. 是，进行融合")
        print("2. 否，仅查看分析结果")
        print("3. 退出程序")

        while True:
            choice = input("\n请选择 (1/2/3): ").strip()
            if choice == "1":
                return True
            elif choice == "2":
                return False
            elif choice == "3":
                print("程序退出")
                sys.exit(0)
            else:
                print("无效选择，请输入 1、2 或 3")

    def get_merge_strategy(self) -> str:
        """选择融合策略"""
        print(f"\n🔧 选择融合策略:")
        print("1. 加权平均 (weighted_average) - 推荐")
        print("   公式: (1-α) × 文件1 + α × 文件2")
        print("2. 相加融合 (add)")
        print("   公式: 文件1 + α × 文件2")
        print("3. 相乘融合 (multiply)")
        print("   公式: 文件1 × (1 + α × 文件2)")

        strategies = {
            "1": "weighted_average",
            "2": "add",
            "3": "multiply"
        }

        while True:
            choice = input("\n请选择融合策略 (1/2/3): ").strip()
            if choice in strategies:
                return strategies[choice]
            print("无效选择，请输入 1、2 或 3")

    def get_alpha_value(self, strategy: str) -> float:
        """获取融合权重α值"""
        print(f"\n⚖️  设置融合权重 α:")

        if strategy == "weighted_average":
            print("α = 0.0: 完全使用文件1")
            print("α = 0.5: 文件1和文件2各占50%")
            print("α = 1.0: 完全使用文件2")
            print("推荐值: 0.3-0.7")
        elif strategy == "add":
            print("α 控制文件2的影响强度")
            print("推荐值: 0.1-0.3")
        elif strategy == "multiply":
            print("α 控制文件2对文件1的调制强度")
            print("推荐值: 0.1-0.5")

        while True:
            try:
                alpha = float(input(f"\n请输入α值 (0.0-1.0): ").strip())
                if 0.0 <= alpha <= 1.0:
                    return alpha
                else:
                    print("α值必须在0.0到1.0之间")
            except ValueError:
                print("请输入有效的数字")

    def get_output_path(self, default_path: str) -> str:
        """获取输出路径"""
        print(f"\n💾 设置输出文件路径:")
        print(f"默认路径: {default_path}")

        choice = input("使用默认路径？(Y/n): ").strip().lower()
        if choice in ['', 'y', 'yes']:
            return default_path

        while True:
            path = input("请输入新的输出路径: ").strip()
            if path:
                # 检查目录是否存在，不存在则创建
                output_dir = os.path.dirname(path)
                if output_dir and not os.path.exists(output_dir):
                    try:
                        os.makedirs(output_dir)
                        print(f"已创建目录: {output_dir}")
                    except Exception as e:
                        print(f"无法创建目录: {e}")
                        continue

                # 检查文件是否存在
                if os.path.exists(path):
                    overwrite = input(f"文件已存在，是否覆盖？(y/N): ").strip().lower()
                    if overwrite not in ['y', 'yes']:
                        continue

                return path

    def load_full_tensors(self, file_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
        """加载完整的张量数据"""
        tensors = {}
        metadata = {}

        print(f"正在加载: {file_path}")
        try:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                metadata = f.metadata() or {}
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
            print(f"✓ 加载完成: {len(tensors)} 个张量")
            return tensors, metadata
        except Exception as e:
            print(f"✗ 加载失败: {str(e)}")
            raise

    def merge_tensors(self, tensors1: Dict[str, torch.Tensor],
                      tensors2: Dict[str, torch.Tensor],
                      alpha: float, strategy: str) -> Dict[str, torch.Tensor]:
        """融合张量"""
        print(f"\n🔄 开始融合...")
        print(f"策略: {strategy}, α = {alpha}")

        keys1 = set(tensors1.keys())
        keys2 = set(tensors2.keys())
        common_keys = keys1.intersection(keys2)

        merged_tensors = {}
        success_count = 0
        skip_count = 0

        # 融合相同的key
        for key in common_keys:
            tensor1 = tensors1[key]
            tensor2 = tensors2[key]

            # 检查兼容性
            if tensor1.shape != tensor2.shape:
                print(f"⚠️  跳过形状不匹配的key: {key}")
                merged_tensors[key] = tensor1
                skip_count += 1
                continue

            if tensor1.dtype != tensor2.dtype:
                tensor2 = tensor2.to(tensor1.dtype)

            try:
                if strategy == "weighted_average":
                    merged_tensor = (1 - alpha) * tensor1 + alpha * tensor2
                elif strategy == "add":
                    merged_tensor = tensor1 + alpha * tensor2
                elif strategy == "multiply":
                    merged_tensor = tensor1 * (1 + alpha * tensor2)

                merged_tensors[key] = merged_tensor
                success_count += 1

            except Exception as e:
                print(f"✗ 融合失败: {key}, 使用文件1的值")
                merged_tensors[key] = tensor1
                skip_count += 1

        # 添加独有的key
        for key in keys1 - keys2:
            merged_tensors[key] = tensors1[key]

        for key in keys2 - keys1:
            merged_tensors[key] = tensors2[key]

        print(f"✓ 融合完成: 成功 {success_count}, 跳过 {skip_count}, 总计 {len(merged_tensors)}")
        return merged_tensors

    def save_merged_file(self, merged_tensors: Dict[str, torch.Tensor], output_path: str):
        """保存融合后的文件"""
        print(f"\n💾 保存文件: {output_path}")

        try:
            # 合并元数据
            merged_metadata = {}
            merged_metadata.update(self.file1_metadata)
            for k, v in self.file2_metadata.items():
                if k not in merged_metadata:
                    merged_metadata[k] = v

            # 添加融合信息
            merged_metadata["merged_by"] = "InteractiveSafeTensorsMerger"
            merged_metadata["merge_timestamp"] = datetime.now().isoformat()
            merged_metadata["source_file1"] = os.path.basename(self.file1_path)
            merged_metadata["source_file2"] = os.path.basename(self.file2_path)

            save_file(merged_tensors, output_path, metadata=merged_metadata)

            # 显示文件信息
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✓ 保存成功!")
            print(f"  文件大小: {file_size:.2f} MB")
            print(f"  张量数量: {len(merged_tensors)}")

        except Exception as e:
            print(f"✗ 保存失败: {str(e)}")
            raise

    def process_files(self, file1_path: str, file2_path: str, default_output_path: str):
        """处理整个交互式融合流程"""
        self.file1_path = file1_path
        self.file2_path = file2_path

        print("=" * 60)
        print("🚀 交互式SafeTensors权重融合工具")
        print("=" * 60)
        print(f"文件1: {file1_path}")
        print(f"文件2: {file2_path}")

        # 检查文件是否存在
        if not os.path.exists(file1_path):
            raise FileNotFoundError(f"文件1不存在: {file1_path}")
        if not os.path.exists(file2_path):
            raise FileNotFoundError(f"文件2不存在: {file2_path}")

        # 加载文件信息进行分析
        print(f"\n📊 正在分析文件...")
        self.file1_info, self.file1_metadata = self.load_safetensors_info(file1_path)
        self.file2_info, self.file2_metadata = self.load_safetensors_info(file2_path)

        # 进行兼容性分析
        analysis = self.analyze_compatibility(self.file1_info, self.file2_info)

        # 显示分析结果
        self.display_analysis_summary(analysis)

        # 询问是否查看详细信息
        show_detail = input(f"\n📋 是否查看详细分析？(y/N): ").strip().lower()
        if show_detail in ['y', 'yes']:
            self.display_detailed_analysis(analysis)

        # 询问是否进行融合
        if not self.get_user_choice_merge():
            print("分析完成，程序结束。")
            return

        # 获取融合参数
        strategy = self.get_merge_strategy()
        alpha = self.get_alpha_value(strategy)
        output_path = self.get_output_path(default_output_path)

        # 确认融合参数
        print(f"\n{'=' * 60}")
        print("📝 融合参数确认:")
        print(f"  策略: {strategy}")
        print(f"  权重α: {alpha}")
        print(f"  输出: {output_path}")
        print(f"  预计融合key数: {len(analysis['compatible_keys'])}")

        confirm = input(f"\n确认开始融合？(Y/n): ").strip().lower()
        if confirm in ['n', 'no']:
            print("操作已取消")
            return

        # 加载完整张量数据
        print(f"\n📦 加载完整数据...")
        file1_tensors, _ = self.load_full_tensors(file1_path)
        file2_tensors, _ = self.load_full_tensors(file2_path)

        # 执行融合
        merged_tensors = self.merge_tensors(file1_tensors, file2_tensors, alpha, strategy)

        # 保存结果
        self.save_merged_file(merged_tensors, output_path)

        print(f"\n🎉 全部完成！")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="交互式SafeTensors权重融合工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python interactive_safetensors_merger.py
  python interactive_safetensors_merger.py --file1 model1.safetensors --file2 model2.safetensors
        """
    )

    parser.add_argument("--file1",
                        default="/dev_share/gdli7/models/pulid/editid_v2_insert_512_5000_id_loss.safetensors",
                        help="第一个safetensors文件路径")
    parser.add_argument("--file2",
                        default="/dev_share/gdli7/models/pulid/editid_v2_insert_512_5000_id_loss_05_10.safetensors",
                        help="第二个safetensors文件路径")
    parser.add_argument("--output",
                        default="/dev_share/gdli7/models/pulid/editid_v2_insert_512_5000_id_loss_05_10_update.safetensors",
                        help="默认输出路径")

    args = parser.parse_args()

    try:
        merger = InteractiveSafeTensorsMerger()
        merger.process_files(args.file1, args.file2, args.output)
    except KeyboardInterrupt:
        print(f"\n\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"✗ 处理失败: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())