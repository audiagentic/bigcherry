"""Configure converter quantizer ISA coverage independently of serving kernels."""
import re
from bigcherry.patcher import Edit, FilePatch

BEFORE = """if(LQ_HIP_SRC)
  target_compile_definitions(libquant PRIVATE LQ_CARD=1)
endif()
"""
AFTER = """if(LQ_HIP_SRC)
  target_compile_definitions(libquant PRIVATE LQ_CARD=1)

  # Converter devices can have architectures absent from serving kernel images.
  # Keep the server default unless the converter requests its own ISA list.
  set(RAD_QUANT_GPU_TARGETS "${RAD_GPU_TARGETS}" CACHE STRING
      "GPU targets for libquant TCQ conversion")
  if(NOT RAD_QUANT_GPU_TARGETS)
    message(FATAL_ERROR "libquant requires RAD_QUANT_GPU_TARGETS")
  endif()
  set_target_properties(libquant PROPERTIES HIP_ARCHITECTURES "${RAD_QUANT_GPU_TARGETS}")
endif()
"""
PATCHES = (FilePatch(
    path="libquant/CMakeLists.txt", language="none",
    description="Converter quantizer HIP code object coverage",
    edits=(Edit(id="iq6r-quantizer-device-arches",
        anchor=re.escape(BEFORE), mode="replace", text=AFTER,
        guard=re.escape("set(RAD_QUANT_GPU_TARGETS"),
        expect_matches=1, max_span_lines=4,
        rationale="Compile the converter trellis kernels for the GPU that executes them"),),
),)
