set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name enzyme)
add_public_tablegen_target(MLIREnzymePassIncGen)
add_mlir_doc(Passes EnzymePasses ./ -gen-pass-doc)

add_mlir_dialect_library(MLIREnzymeTransforms
    EnzymeMLIRPass.cpp
    EnzymeBatchPass.cpp
    EnzymeWrapPass.cpp
    PrintActivityAnalysis.cpp
    PrintAliasAnalysis.cpp
    EnzymeToMemRef.cpp
    SimplifyMath.cpp
    AddToOpToIndexAndLoad.cpp
    AddToOpToSplit.cpp
    RemoveUnusedEnzymeOps.cpp
    SimplifyMemrefCache.cpp
    Utils.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIREnzymePassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRAffineUtils
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMathToLLVM
  MLIRMemRefDialect
  MLIRNVVMDialect
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRControlFlowInterfaces
  MLIRSCFToControlFlow
  MLIRTransformUtils

  MLIREnzymeAutoDiffInterface
)
