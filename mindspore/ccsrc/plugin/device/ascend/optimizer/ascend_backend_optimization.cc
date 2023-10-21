/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "plugin/device/ascend/optimizer/ascend_backend_optimization.h"
#include <memory>
#include <string>
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/debug/profiler/profiling.h"
#include "plugin/device/ascend/optimizer/ir_fission/dynamic_rnn_grad_fission_v2.h"
#include "plugin/device/ascend/optimizer/ir_fission/dynamic_gru_v2_grad_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/bn_split.h"
#include "plugin/device/ascend/optimizer/ir_fission/bn_grad_split.h"
#include "plugin/device/ascend/optimizer/ir_fission/single_batch_norm_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/reduce_min_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/layer_norm_grad_split.h"
#include "plugin/device/ascend/optimizer/ir_fusion/unsorted_segment_sum_replace.h"
#include "plugin/device/ascend/optimizer/ir_fission/unsorted_segment_sum_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/unsorted_segment_sum_d_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/gather_v2_ds_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/bce_with_logits_loss_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/broadcastto_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/reduce_sum_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/add_status_input_for_random_operator.h"
#include "plugin/device/ascend/optimizer/ir_fission/renorm_split.h"
#include "plugin/device/ascend/optimizer/ir_fission/resize_linear1d_fission.h"
#include "backend/common/pass/communication_op_fusion.h"
#include "backend/common/pass/dropout_gen_mask_fusion.h"
#include "backend/common/pass/dynamic_sequence_ops_adaptation.h"
#include "backend/common/pass/merge_transdata.h"
#include "plugin/device/ascend/optimizer/ir_fusion/square_sum_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/clip_by_norm_no_div_square_sum_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/lamb_update_with_lr_rule_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/prelu_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/clip_by_value_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/confusion_softmax_grad_rule.h"
#include "plugin/device/ascend/optimizer/ir_fusion/lamb_next_mv_rule.h"
#include "plugin/device/ascend/optimizer/ir_fusion/lamb_next_mv_with_decay_rule.h"
#include "plugin/device/ascend/optimizer/ir_fusion/lamb_next_right_rule.h"
#include "plugin/device/ascend/optimizer/ir_fusion/lamb_update_with_lr_v2.h"
#include "plugin/device/ascend/optimizer/ir_fusion/layer_norm_beta_gamma_backprop_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/reshape_transpose_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/transpose_reshape_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/adam_apply_one_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/adam_apply_one_with_decay_rule.h"
#include "plugin/device/ascend/optimizer/ir_fusion/parameter_and_transop_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/refresh_parameter_format.h"
#include "plugin/device/ascend/optimizer/ir_fusion/transpose_transdata_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/deformable_offsets_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/deformable_offsets_grad_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fission/transdata_split.h"
#include "plugin/device/ascend/optimizer/ir_fission/topk_split.h"
#include "plugin/device/ascend/optimizer/ir_fission/conv2d_backprop_filter_mul_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/space_to_depth_split.h"
#include "plugin/device/ascend/optimizer/ir_fission/max_pool3d_grad_grad_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/scale_grad_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/maximum_grad_fission.h"
#include "plugin/device/ascend/optimizer/ir_fusion/adaptive_max_pool2d_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/avgpool_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/avgpool_3d_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/avgpool_3d_grad_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/momentum_lossscale_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/mul_add_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/mul_addn_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/matmul_biasadd_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/conv2d_backprop_input_biasadd_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/remove_reshape_pair.h"
#include "plugin/device/ascend/optimizer/ir_fusion/derelu_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/confusion_mul_grad_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/softmax_grad_ext_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/bn_reduce_grad_conv2d_backprop_filter_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/transposed_update_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/softmax_dropout_do_mask_v3_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/conv2d_backprop_input_dilation_fusion.h"
#include "plugin/device/ascend/optimizer/format_type/insert_trans_op.h"
#include "plugin/device/ascend/optimizer/format_type/reselect_call_inline_format.h"
#include "plugin/device/ascend/optimizer/format_type/trans_op_format_refine.h"
#include "plugin/device/ascend/optimizer/format_type/dynamic_rnn_grad_reformat.h"
#include "plugin/device/ascend/optimizer/format_type/insert_transpose_for_basiclstm_op.h"
#include "plugin/device/ascend/optimizer/format_type/insert_transpose_for_dyanmic_gru_v2.h"
#include "plugin/device/ascend/optimizer/format_type/rectify_do_mask_kernel_info.h"
#include "plugin/device/ascend/optimizer/format_type/change_axis_of_reduce_kernel.h"
#include "plugin/device/ascend/optimizer/format_type/convert_cast_format.h"
#include "plugin/device/ascend/optimizer/format_type/set_fracz_group_attr.h"
#include "plugin/device/ascend/optimizer/mindir/aicpu_lib_select.h"
#include "plugin/device/ascend/optimizer/ir_fission/seed_adapter.h"
#include "backend/common/pass/getitem_tuple.h"
#include "backend/common/pass/optimize_dependence.h"
#include "backend/common/pass/erase_visit_attr.h"
#include "plugin/device/ascend/optimizer/format_type/insert_cast.h"
#include "plugin/device/ascend/optimizer/format_type/convert_unsupported_transnode_to_aicpu.h"
#include "backend/common/pass/eliminate_redundant_op.h"
#include "backend/common/pass/common_subexpression_elimination.h"
#include "plugin/device/ascend/optimizer/format_type/merge_cast_to_op.h"
#include "plugin/device/ascend/optimizer/format_type/remove_host_kernel.h"
#include "plugin/device/ascend/optimizer/format_type/check_consistency.h"
#include "plugin/device/ascend/optimizer/format_type/eliminate_graph_output_transdata.h"
#include "plugin/device/ascend/optimizer/format_type/deal_ref_output.h"
#include "plugin/device/ascend/optimizer/enhancer/skip_empty_tensor_output.h"
#include "plugin/device/ascend/optimizer/enhancer/insert_tensor_move_for_hccl_op.h"
#include "plugin/device/ascend/optimizer/enhancer/insert_tensor_move_for_cascade.h"
#include "plugin/device/ascend/optimizer/enhancer/insert_pad_for_nms_with_mask.h"
#include "plugin/device/ascend/optimizer/enhancer/transpose_optimizer.h"
#include "plugin/device/ascend/optimizer/format_type/insert_transdata_for_runop.h"
#include "plugin/device/ascend/optimizer/enhancer/insert_transpose_for_sort.h"
#include "plugin/device/ascend/optimizer/enhancer/getnext_tensor_move_elimination.h"
#include "plugin/device/ascend/optimizer/ir_fission/addn_fission.h"
#include "plugin/device/ascend/optimizer/enhancer/insert_tensor_move_for_getnext.h"
#include "plugin/device/ascend/optimizer/ir_fission/split_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/splitv_fission.h"
#include "plugin/device/ascend/optimizer/format_type/remove_internal_output.h"
#include "plugin/device/ascend/optimizer/ir_fission/concat_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/pack_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/lamb_fission.h"
#include "plugin/device/ascend/optimizer/enhancer/concat_outputs_for_all_gather.h"
#include "plugin/device/ascend/optimizer/enhancer/insert_depend_for_all_gather.h"
#include "plugin/device/ascend/optimizer/enhancer/insert_depend_for_grad_comm.h"
#include "plugin/device/ascend/optimizer/enhancer/insert_depend_for_all_gather_output.h"
#include "plugin/device/ascend/optimizer/enhancer/insert_depend_for_all_reduce.h"
#include "plugin/device/ascend/optimizer/enhancer/split_inputs_for_reduce_scatter.h"
#include "plugin/device/ascend/optimizer/enhancer/add_attr_for_3d_graph.h"
#include "plugin/device/ascend/optimizer/enhancer/split_n_optimizer.h"
#include "plugin/device/ascend/optimizer/enhancer/eliminate_maketuple_getitem.h"
#include "plugin/device/ascend/optimizer/mindir/maxpool_to_maxpool_with_argmax.h"
#include "plugin/device/ascend/optimizer/mindir/maxpool_with_argmax_unify_mindir.h"
#include "plugin/device/ascend/optimizer/mindir/dropout_unify_mindir.h"
#include "plugin/device/ascend/optimizer/mindir/optimizer_unify_output.h"
#include "plugin/device/ascend/optimizer/mindir/sparse_softmax_cross_entropy_with_logits_unify_mindir.h"
#include "plugin/device/ascend/optimizer/mindir/slice_grad_unify_mindir.h"
#include "plugin/device/ascend/optimizer/mindir/update_input_names_strided_slice_grad.h"
#include "plugin/device/ascend/optimizer/mindir/ascend_vm_op_adapter.h"
#include "plugin/device/ascend/optimizer/ir_fusion/padd_update_fusion.h"
#include "backend/common/pass/adjust_depend_for_parallel_optimizer_recompute_all_gather.h"
#include "backend/common/pass/gradients_allreduce_depend_last_send.h"
#include "backend/common/pass/optimize_gradients_allreduce_overlap.h"
#include "include/common/utils/config_manager.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/dump_proto.h"
#include "include/common/debug/draw.h"
#include "plugin/device/ascend/optimizer/optimizer_factory.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/optimizer/format_type/replace_transdata_with_transpose.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/ub_fusion_optimizer.h"
#include "plugin/device/ascend/optimizer/backend_common_unify_mindir.h"

namespace mindspore {
namespace opt {
namespace {
void AddAscendIRFusionRulesPass(PassManager *ir_fusion_pm) {
  MS_EXCEPTION_IF_NULL(ir_fusion_pm);
  ir_fusion_pm->AddPass(std::make_shared<LambUpdateWithLRRuleFusion>());
  ir_fusion_pm->AddPass(std::make_shared<LambNextMVWithDecayRuleCond1>());
  ir_fusion_pm->AddPass(std::make_shared<LambNextMVWithDecayRuleCond2>());
  ir_fusion_pm->AddPass(std::make_shared<LambNextMVWithDecayRuleCond3>());
  ir_fusion_pm->AddPass(std::make_shared<LambNextMVWithDecayRuleCond4>());
  ir_fusion_pm->AddPass(std::make_shared<LambNextMVRuleCond1>());
  ir_fusion_pm->AddPass(std::make_shared<LambNextMVRuleCond2>());
  ir_fusion_pm->AddPass(std::make_shared<LambNextMVRuleCond3>());
  ir_fusion_pm->AddPass(std::make_shared<LambNextMVRuleCond4>());
  ir_fusion_pm->AddPass(std::make_shared<LambNextRightRule>());
  ir_fusion_pm->AddPass(std::make_shared<LambUpdateWithLrV2>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneCond1Fusion>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneCond2Fusion>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneCond3Fusion>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneCond4Fusion>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneWithDecayRuleCond1>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneWithDecayRuleCond2>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneWithDecayRuleCond3>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneWithDecayRuleCond4>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneWithDecayRuleCond5>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneWithDecayRuleCond6>());
  ir_fusion_pm->AddPass(std::make_shared<ClipByNormNoDivSquareSumFusion>());
  ir_fusion_pm->AddPass(std::make_shared<SquareSumFusion>());
  ir_fusion_pm->AddPass(std::make_shared<ClipByValueFusion>());
  ir_fusion_pm->AddPass(std::make_shared<PReluFusion>());
}

void AddAscendIRFusionPass(PassManager *ir_fusion_pm) {
  MS_EXCEPTION_IF_NULL(ir_fusion_pm);
  ir_fusion_pm->AddPass(std::make_shared<UnsortedSegmentSumReplace>());
  ir_fusion_pm->AddPass(std::make_shared<SingleBatchNormFission>());
  ir_fusion_pm->AddPass(std::make_shared<GetitemTuple>());
  ir_fusion_pm->AddPass(std::make_shared<SoftmaxGradExtFusion>());
  ir_fusion_pm->AddPass(std::make_shared<SoftmaxGradExtFusionV2>());
  ir_fusion_pm->AddPass(std::make_shared<SoftmaxGradExtFusionV3>());
  ir_fusion_pm->AddPass(std::make_shared<ConfusionMulGradFusion>());
  ir_fusion_pm->AddPass(std::make_shared<ConfusionSoftmaxGradRule>());
  ir_fusion_pm->AddPass(std::make_shared<ReshapeTransposeFusion>());
  ir_fusion_pm->AddPass(std::make_shared<TransposeReshapeFusion>());
  ir_fusion_pm->AddPass(std::make_shared<Conv2dBackpropFilterMul>());
  ir_fusion_pm->AddPass(std::make_shared<DeformableOffsetsFusion>());
  ir_fusion_pm->AddPass(std::make_shared<DeformableOffsetsGradFusion>());
  ir_fusion_pm->AddPass(std::make_shared<MaxPool3DGradGradFission>());
  ir_fusion_pm->AddPass(std::make_shared<AdaptiveMaxPool2DFusion>());
  ir_fusion_pm->AddPass(std::make_shared<AvgPoolFusion>());
  ir_fusion_pm->AddPass(std::make_shared<AvgPool3DFusion>());
  ir_fusion_pm->AddPass(std::make_shared<AvgPool3DGradFusion>());
  ir_fusion_pm->AddPass(std::make_shared<MomentumLossscaleFusion>());
  ir_fusion_pm->AddPass(std::make_shared<MulAddFusion>());
  ir_fusion_pm->AddPass(std::make_shared<MulAddNFusion>());
  ir_fusion_pm->AddPass(std::make_shared<MatmulBiasaddFusion>());
  ir_fusion_pm->AddPass(std::make_shared<Conv2dBackpropInputBiasaddFusion>());
  ir_fusion_pm->AddPass(std::make_shared<MatmulAddFusion>());
  ir_fusion_pm->AddPass(std::make_shared<AddnFission>());
  ir_fusion_pm->AddPass(std::make_shared<DereluFusion>());
  ir_fusion_pm->AddPass(std::make_shared<TransposeTransDataFusion>());
  ir_fusion_pm->AddPass(std::make_shared<DynamicRnnGradFissionV2>());
  ir_fusion_pm->AddPass(std::make_shared<SplitFission>());
  ir_fusion_pm->AddPass(std::make_shared<SplitVFission>());
  ir_fusion_pm->AddPass(std::make_shared<SpaceToDepthSplit>());
  ir_fusion_pm->AddPass(std::make_shared<GetitemTuple>());
  ir_fusion_pm->AddPass(std::make_shared<PackFission>());
  ir_fusion_pm->AddPass(std::make_shared<ConcatFission>());
  ir_fusion_pm->AddPass(std::make_shared<ReduceMinFission>());
  ir_fusion_pm->AddPass(std::make_shared<UnsortedSegmentSumFission>());
  ir_fusion_pm->AddPass(std::make_shared<UnsortedSegmentSumDFission>());
  ir_fusion_pm->AddPass(std::make_shared<GatherV2DsFission>());
  ir_fusion_pm->AddPass(std::make_shared<BCEWithLogitsLossFission>());
  ir_fusion_pm->AddPass(std::make_shared<BroadcasttoFission>());
  ir_fusion_pm->AddPass(std::make_shared<DynamicBroadcastToFission>());
  ir_fusion_pm->AddPass(std::make_shared<ReduceSumFission>());
  ir_fusion_pm->AddPass(std::make_shared<BNReduceGradConv2dBackpropFilterFusion>());
  ir_fusion_pm->AddPass(std::make_shared<SoftmaxDropoutDoMaskV3Fusion>());
  const auto &pass_creators =
    opt::Factory<PatternProcessPass>::Instance().GetPassCreatorsByType(kPassType::kIRFusionFissionPass);
  for (const auto &pass_creator : pass_creators) {
    ir_fusion_pm->AddPass(pass_creator.second());
  }
  const auto &pattern_to_pattern_creators =
    opt::Factory<PatternToPatternPass>::Instance().GetPassCreatorsByType(kPassType::kIRFusionFissionPass);
  for (const auto &pattern_to_pattern_pass : pattern_to_pattern_creators) {
    ir_fusion_pm->AddPass(pattern_to_pattern_pass.second());
  }
}
}  // namespace

void AscendDataLayout(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_AscendDataLayout", 0, 0, 0);
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto data_layout_pm = std::make_shared<PassManager>("transop_pm");
  data_layout_pm->AddPass(std::make_shared<ReselectCallInlineFormat>());
  data_layout_pm->AddPass(std::make_shared<RectifyDoMaskKernelInfo>());
  data_layout_pm->AddPass(std::make_shared<DynamicRNNGradReformat>());
  data_layout_pm->AddPass(std::make_shared<ChangeAxisOfReduceKernel>());
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode &&
      !kernel_graph->has_flag(kFlagPyNativeRunInGraph)) {
    data_layout_pm->AddPass(std::make_shared<RunOpInsertTransData>());
  } else {
    data_layout_pm->AddPass(std::make_shared<MergeCastToOp>());
    data_layout_pm->AddPass(std::make_shared<ConvertCastFormat>());
    data_layout_pm->AddPass(std::make_shared<EraseVisitAttr>());
    data_layout_pm->AddPass(std::make_shared<InsertTransOp>());
    data_layout_pm->AddPass(std::make_shared<GetitemTuple>());
  }
  data_layout_pm->AddPass(std::make_shared<EraseVisitAttr>());
  data_layout_pm->AddPass(std::make_shared<AddIoFormatAttrFor3DGraph>());
  data_layout_pm->AddPass(std::make_shared<CommonSubexpressionElimination>());
  data_layout_pm->AddPass(std::make_shared<RemoveReshapePair>());
  data_layout_pm->AddPass(std::make_shared<EliminateRedundantOp>());
  data_layout_pm->AddPass(std::make_shared<ReplaceTransDataWithTranspose>());
  if (kernel_graph->has_flag(kFlagPyNativeRunInGraph)) {
    data_layout_pm->AddPass(std::make_shared<EliminateGraphOutputTransdata>());
  }
  data_layout_pm->AddPass(std::make_shared<InsertTransposeForDynamicGRUV2>());
  data_layout_pm->AddPass(std::make_shared<OptimizeDependence>());
  data_layout_pm->AddPass(std::make_shared<TransDataSplit>());
  data_layout_pm->AddPass(std::make_shared<EraseVisitAttr>());
  data_layout_pm->AddPass(std::make_shared<RemoveInternalOutputTransOp>());
  optimizer->AddPassManager(data_layout_pm);
  (void)optimizer->Optimize(kernel_graph);
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_AscendDataLayout", 0, 0, 1);
  kernel_graph->SetExecOrderByDefault();
}

void AscendMixPrecision(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_AscendMixPrecision", 0, 0, 0);
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto mixed_precision_pm = std::make_shared<PassManager>("cast_pm");
  mixed_precision_pm->AddPass(std::make_shared<InsertCast>());
  mixed_precision_pm->AddPass(std::make_shared<GetitemTuple>());
  mixed_precision_pm->AddPass(std::make_shared<CommonSubexpressionElimination>());
  mixed_precision_pm->AddPass(std::make_shared<EliminateRedundantOp>());
  mixed_precision_pm->AddPass(std::make_shared<OptimizeDependence>());
  mixed_precision_pm->AddPass(std::make_shared<EraseVisitAttr>());
  mixed_precision_pm->AddPass(std::make_shared<DealRefOutput>());
  mixed_precision_pm->AddPass(std::make_shared<TransDataSplit>());
  mixed_precision_pm->AddPass(std::make_shared<GetitemTuple>());
  mixed_precision_pm->AddPass(std::make_shared<MergeCastToOp>());
  mixed_precision_pm->AddPass(std::make_shared<LayerNormBetaGammaBackpropFusion>());
  mixed_precision_pm->AddPass(std::make_shared<EraseVisitAttr>());
  mixed_precision_pm->AddPass(std::make_shared<TransOpFormatRefine>());
  mixed_precision_pm->AddPass(std::make_shared<EraseVisitAttr>());
  mixed_precision_pm->AddPass(std::make_shared<TransposedUpdateFusion>());
  mixed_precision_pm->AddPass(std::make_shared<TransposeOptimizer>());
  mixed_precision_pm->AddPass(std::make_shared<ConvertUnSupportNodeToAICPU>());
  mixed_precision_pm->AddPass(std::make_shared<RemoveInternalOutputCast>());
  optimizer->AddPassManager(mixed_precision_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_AscendMixPrecision", 0, 0, 1);
}

void AscendBackendIRFusionOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Status record: start ascend ir fusion pass. graph id: " << kernel_graph->graph_id();
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_IRFusionOptimization", 0, 0, 0);
  PROF_START(ir_fusion);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_ir_fusion_before_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
    DumpIRProto(kernel_graph, "before_hwopt_" + std::to_string(kernel_graph->graph_id()));
  }
#endif
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto ir_fusion_pm = std::make_shared<PassManager>("ir_fusion_pm");
  ir_fusion_pm->AddPass(std::make_shared<opt::LambFission>());
  ir_fusion_pm->AddPass(std::make_shared<opt::MaxPool2MaxPoolWithArgmax>());
  ir_fusion_pm->AddPass(std::make_shared<opt::MaxPoolWithArgmaxUnifyMindIR>());
  ir_fusion_pm->AddPass(std::make_shared<opt::MaxPoolGradWithArgmaxUnifyMindIR>());
  ir_fusion_pm->AddPass(std::make_shared<opt::SliceGradUnifyMindIR>());
  ir_fusion_pm->AddPass(std::make_shared<opt::StridedSliceGradUpdateInputNames>());
  ir_fusion_pm->AddPass(std::make_shared<SeedAdapter>());
  ir_fusion_pm->AddPass(std::make_shared<AddStatusInputForRandomOperator>());
  ir_fusion_pm->AddPass(std::make_shared<EraseVisitAttr>());
  ir_fusion_pm->AddPass(std::make_shared<SyncBnSplit>());
  ir_fusion_pm->AddPass(std::make_shared<SyncBnGradSplit>());
  ir_fusion_pm->AddPass(std::make_shared<Conv2dBackpropInputDilationFusion>());
  ir_fusion_pm->AddPass(std::make_shared<LayerNormGradSplit>());
  ir_fusion_pm->AddPass(std::make_shared<ScaleGradFission>());
  if (kernel_graph->is_dynamic_shape()) {
    ir_fusion_pm->AddPass(std::make_shared<MaximumGradFission>());
  }
  ir_fusion_pm->AddPass(std::make_shared<InsertPadForNMSWithMask>());
  ir_fusion_pm->AddPass(std::make_shared<DynamicGRUV2GradFission>());
  ir_fusion_pm->AddPass(std::make_shared<InsertTransposeForSort>());
  ir_fusion_pm->AddPass(std::make_shared<EraseVisitAttr>());
  ir_fusion_pm->AddPass(std::make_shared<RenormSplit>());
  ir_fusion_pm->AddPass(std::make_shared<EraseVisitAttr>());
  ir_fusion_pm->AddPass(std::make_shared<RemoveHostKernel>());
  ir_fusion_pm->AddPass(std::make_shared<ResizeLinear1DFission>());
  ir_fusion_pm->AddPass(std::make_shared<ResizeLinear1DGradFission>());
  AddAscendIRFusionRulesPass(ir_fusion_pm.get());
  AddAscendIRFusionPass(ir_fusion_pm.get());

  if (context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK) && context_ptr->get_param<bool>(MS_CTX_ENABLE_LOOP_SINK) &&
      ConfigManager::GetInstance().iter_num() > 1) {
    ir_fusion_pm->AddPass(std::make_shared<InsertTensorMoveForGetNext>());
    ir_fusion_pm->AddPass(std::make_shared<GetitemTuple>());
    ir_fusion_pm->AddPass(std::make_shared<EraseVisitAttr>());
  }
  ir_fusion_pm->AddPass(std::make_shared<CommonSubexpressionElimination>());
  ir_fusion_pm->AddPass(std::make_shared<InsertTensorMoveForHcclOp>());
  ir_fusion_pm->AddPass(std::make_shared<InsertTranspose>());
  ir_fusion_pm->AddPass(std::make_shared<GetitemTuple>());
  ir_fusion_pm->AddPass(std::make_shared<EraseVisitAttr>());
  ir_fusion_pm->AddPass(std::make_shared<SkipEmptyTensorOutputPass>());
  optimizer->AddPassManager(ir_fusion_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_ir_fusion_after_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  PROF_END(ir_fusion);
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_IRFusionOptimization", 0, 0, 1);
  MS_LOG(INFO) << "Status record: end ascend ir fusion pass. graph id: " << kernel_graph->graph_id();
}

void RunOpAscendBackendIRFusionOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->get_param<bool>(MS_CTX_IR_FUSION_FLAG)) {
    MS_LOG(INFO) << "IRFusion is not enable, skip";
    return;
  }
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_RunOpIRFusionOptimization", 0, 0, 0);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    DumpIR("hwopt_d_ir_fusion_before.ir", kernel_graph);
  }
#endif
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto ir_fusion_pm = std::make_shared<PassManager>("ir_fusion_pm");
  ir_fusion_pm->AddPass(std::make_shared<opt::LambFission>());
  ir_fusion_pm->AddPass(std::make_shared<UnsortedSegmentSumReplace>());
  ir_fusion_pm->AddPass(std::make_shared<DynamicGRUV2GradFission>());
  ir_fusion_pm->AddPass(std::make_shared<DynamicRnnGradFissionV2>());
  ir_fusion_pm->AddPass(std::make_shared<SplitFission>());
  ir_fusion_pm->AddPass(std::make_shared<SplitVFission>());
  ir_fusion_pm->AddPass(std::make_shared<ConcatFission>());
  ir_fusion_pm->AddPass(std::make_shared<SeedAdapter>());
  ir_fusion_pm->AddPass(std::make_shared<AddStatusInputForRandomOperator>());
  ir_fusion_pm->AddPass(std::make_shared<EraseVisitAttr>());
  ir_fusion_pm->AddPass(std::make_shared<SyncBnSplit>());
  ir_fusion_pm->AddPass(std::make_shared<SyncBnGradSplit>());
  ir_fusion_pm->AddPass(std::make_shared<Conv2dBackpropInputDilationFusion>());
  ir_fusion_pm->AddPass(std::make_shared<LayerNormGradSplit>());
  ir_fusion_pm->AddPass(std::make_shared<Conv2dBackpropFilterMul>());
  ir_fusion_pm->AddPass(std::make_shared<SpaceToDepthSplit>());
  ir_fusion_pm->AddPass(std::make_shared<DeformableOffsetsFusion>());
  ir_fusion_pm->AddPass(std::make_shared<DeformableOffsetsGradFusion>());
  ir_fusion_pm->AddPass(std::make_shared<ScaleGradFission>());
  ir_fusion_pm->AddPass(std::make_shared<MaxPool3DGradGradFission>());
  ir_fusion_pm->AddPass(std::make_shared<AdaptiveMaxPool2DFusion>());
  ir_fusion_pm->AddPass(std::make_shared<AvgPoolFusion>());
  ir_fusion_pm->AddPass(std::make_shared<AvgPool3DFusion>());
  ir_fusion_pm->AddPass(std::make_shared<AvgPool3DGradFusion>());
  ir_fusion_pm->AddPass(std::make_shared<AddnFission>());
  ir_fusion_pm->AddPass(std::make_shared<InsertPadForNMSWithMask>());
  ir_fusion_pm->AddPass(std::make_shared<EraseVisitAttr>());
  ir_fusion_pm->AddPass(std::make_shared<BroadcasttoFission>());
  ir_fusion_pm->AddPass(std::make_shared<DynamicBroadcastToFission>());
  ir_fusion_pm->AddPass(std::make_shared<ReduceSumFission>());
  ir_fusion_pm->AddPass(std::make_shared<BCEWithLogitsLossFission>());
  ir_fusion_pm->AddPass(std::make_shared<InsertTensorMoveForHcclOp>());
  ir_fusion_pm->AddPass(std::make_shared<InsertTransposeForSort>());
  ir_fusion_pm->AddPass(std::make_shared<EraseVisitAttr>());
  ir_fusion_pm->AddPass(std::make_shared<RenormSplit>());
  ir_fusion_pm->AddPass(std::make_shared<EraseVisitAttr>());
  ir_fusion_pm->AddPass(std::make_shared<RemoveHostKernel>());
  ir_fusion_pm->AddPass(std::make_shared<PackFission>());
  ir_fusion_pm->AddPass(std::make_shared<ResizeLinear1DFission>());
  ir_fusion_pm->AddPass(std::make_shared<ResizeLinear1DGradFission>());
  ir_fusion_pm->AddPass(std::make_shared<opt::MaxPoolWithArgmaxUnifyMindIR>());
  ir_fusion_pm->AddPass(std::make_shared<SkipEmptyTensorOutputPass>());
  const auto &pass_creators =
    opt::Factory<PatternProcessPass>::Instance().GetPassCreatorsByType(kPassType::kIRFusionFissionPass);
  for (const auto &pass_creator : pass_creators) {
    ir_fusion_pm->AddPass(pass_creator.second());
  }
  const auto &pattern_to_pattern_creators =
    opt::Factory<PatternToPatternPass>::Instance().GetPassCreatorsByType(kPassType::kIRFusionFissionPass);
  for (const auto &pattern_to_pattern_pass : pattern_to_pattern_creators) {
    ir_fusion_pm->AddPass(pattern_to_pattern_pass.second());
  }
  AddAscendIRFusionRulesPass(ir_fusion_pm.get());

  optimizer->AddPassManager(ir_fusion_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    DumpIR("hwopt_d_ir_fusion_after.ir", kernel_graph);
  }
#endif
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_RunOpIRFusionOptimization", 0, 0, 1);
}

void RunOpAscendBackendOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_RunOpOptimization", 0, 0, 0);
  // data layout optimization
  AscendDataLayout(kernel_graph);
  // mixed precision optimization
  AscendMixPrecision(kernel_graph);
  // other optimization
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto other_pm = std::make_shared<PassManager>("other_pm");
  other_pm->AddPass(std::make_shared<SetFraczGroupAttr>());
  other_pm->AddPass(std::make_shared<PaddUpdateFusion>());
  other_pm->AddPass(std::make_shared<opt::AICpuLibSelectPass>());
  optimizer->AddPassManager(other_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_RunOpOptimization", 0, 0, 1);
}

void AscendAfterInlineOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_AfterInlineOptimization", 0, 0, 0);
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto after_inline_pm = std::make_shared<PassManager>("after_inline_pm");
  after_inline_pm->AddPass(std::make_shared<DropoutGenMaskFusion>());
  after_inline_pm->AddPass(std::make_shared<CommonSubexpressionElimination>());
  after_inline_pm->AddPass(std::make_shared<EliminateRedundantOp>());
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const auto cell_reuse = context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  if (cell_reuse) {
    after_inline_pm->AddPass(std::make_shared<EliminateMaketupleGetitem>());
    after_inline_pm->AddPass(std::make_shared<AllReduceFusion>());
    after_inline_pm->AddPass(std::make_shared<AdjustDependForParallelOptimizerRecomputeAllGather>());
    after_inline_pm->AddPass(std::make_shared<AllGatherFusion>());
    after_inline_pm->AddPass(std::make_shared<ConcatOutputsForAllGather>());
    after_inline_pm->AddPass(std::make_shared<InsertDependForAllGather>());
    after_inline_pm->AddPass(std::make_shared<ReduceScatterFusion>());
    after_inline_pm->AddPass(std::make_shared<SplitInputsForReduceScatter>());
    after_inline_pm->AddPass(std::make_shared<BroadcastFusion>());
    after_inline_pm->AddPass(std::make_shared<InsertTensorMoveForCascade>());
    after_inline_pm->AddPass(std::make_shared<InsertDependForGradComm>());
    after_inline_pm->AddPass(std::make_shared<GradientsAllReduceDependLastSend>());
  }
  after_inline_pm->AddPass(std::make_shared<MergeTransData>());
  optimizer->AddPassManager(after_inline_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_AfterInlineOptimization", 0, 0, 1);
}

void AscendBackendOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Status record: start ascend backend(data layer & mix precision ...) pass. graph id: "
               << kernel_graph->graph_id();
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_Optimization", 0, 0, 0);
  PROF_START(ascend_backend_optimization);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_before_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  // data layout optimization
  AscendDataLayout(kernel_graph);
  // mixed precision optimization
  AscendMixPrecision(kernel_graph);
  // other optimization
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto other_pm = std::make_shared<PassManager>("other_pm");
  other_pm->AddPass(std::make_shared<SendFusion>());
  other_pm->AddPass(std::make_shared<RecvFusion>());
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const auto no_cell_reuse = context->CellReuseLevel() == CellReuseLevel::kNoCellReuse;
  if (no_cell_reuse) {
    other_pm->AddPass(std::make_shared<OptimizeGradientsAllReduceOverlap>());
    other_pm->AddPass(std::make_shared<AllReduceFusion>());
    other_pm->AddPass(std::make_shared<InsertDependForAllReduce>());
    other_pm->AddPass(std::make_shared<AdjustDependForParallelOptimizerRecomputeAllGather>());
    other_pm->AddPass(std::make_shared<AllGatherFusion>());
    other_pm->AddPass(std::make_shared<ConcatOutputsForAllGather>());
    other_pm->AddPass(std::make_shared<InsertDependForAllGatherOutput>());
    other_pm->AddPass(std::make_shared<InsertDependForAllGather>());
    other_pm->AddPass(std::make_shared<ReduceScatterFusion>());
    other_pm->AddPass(std::make_shared<SplitInputsForReduceScatter>());
    other_pm->AddPass(std::make_shared<BroadcastFusion>());
    other_pm->AddPass(std::make_shared<InsertTensorMoveForCascade>());
    other_pm->AddPass(std::make_shared<InsertDependForGradComm>());
    other_pm->AddPass(std::make_shared<GradientsAllReduceDependLastSend>());
  }
  other_pm->AddPass(std::make_shared<ParameterTransOpFusion>());
  other_pm->AddPass(std::make_shared<RefreshParameterFormat>());
  other_pm->AddPass(std::make_shared<SplitOpOptimizer>());
  other_pm->AddPass(std::make_shared<SetFraczGroupAttr>());
  other_pm->AddPass(std::make_shared<PaddUpdateFusion>());
  other_pm->AddPass(std::make_shared<opt::AICpuLibSelectPass>());
  optimizer->AddPassManager(other_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  // buffer fusion
  AscendBackendUBFusionOptimization(kernel_graph);

  // other2 optimization
  auto optimizer2 = std::make_shared<GraphOptimizer>();
  auto other2_pm = std::make_shared<PassManager>("other2_pm");
  other2_pm->AddPass(std::make_shared<GetitemTuple>());
  other2_pm->AddPass(std::make_shared<CommonSubexpressionElimination>());
  if (context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK) && context_ptr->get_param<bool>(MS_CTX_ENABLE_LOOP_SINK) &&
      ConfigManager::GetInstance().iter_num() > 1) {
    other2_pm->AddPass(std::make_shared<GetnextTensorMoveElimination>());
  }
  other2_pm->AddPass(std::make_shared<CheckConsistency>());
  optimizer2->AddPassManager(other2_pm);
  (void)optimizer2->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_end_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
    DumpIRProto(kernel_graph, "after_hwopt_" + std::to_string(kernel_graph->graph_id()));
    if (context_ptr->CanDump(kFully)) {
      draw::Draw("hwopt_d_end.dot", kernel_graph);
    }
  }
#endif
  PROF_END(ascend_backend_optimization);
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_Optimization", 0, 0, 1);
  MS_LOG(INFO) << "Status record: end ascend backend(data layer & mix precision ...) pass. graph id: "
               << kernel_graph->graph_id();
}

PassManagerPtr GetAscendUnifyMindIRPassManager() {
  auto unify_mindir_pm = std::make_shared<opt::PassManager>("ascend_unify_mindir_pm");
  GetBackendCommonUnifyMindIRPassManager(&unify_mindir_pm);
  return unify_mindir_pm;
}

void AscendUnifyMindIR(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_UnifyMindIR", 0, 0, 0);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_before_unify_mindir_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
    DumpIRProto(kernel_graph, "before_unify_mindir_hwopt_" + std::to_string(kernel_graph->graph_id()));
  }
#endif
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  optimizer->AddPassManager(GetAscendUnifyMindIRPassManager());
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_after_unify_mindir_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_UnifyMindIR", 0, 0, 1);
}

void AscendOpAdaptation(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_OpAdaptation", 0, 0, 0);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_before_op_adaptation_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
    DumpIRProto(kernel_graph, "before_op_adaptation_hwopt_" + std::to_string(kernel_graph->graph_id()));
  }
#endif
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto op_adaptation_pm = std::make_shared<opt::PassManager>("op_adaptation_pm");
  op_adaptation_pm->AddPass(std::make_shared<opt::DynamicSequenceOpsAdaptation>());
  op_adaptation_pm->AddPass(std::make_shared<opt::AscendVmOpAdapter>());

  optimizer->AddPassManager(op_adaptation_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_after_op_adaptation_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_OpAdaptation", 0, 0, 1);
}
}  // namespace opt
}  // namespace mindspore
