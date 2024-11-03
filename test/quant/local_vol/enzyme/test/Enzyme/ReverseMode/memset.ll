; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg)" -S | FileCheck %s

declare void @__enzyme_autodiff(i8*, double*, double*, double*, double*)

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)

define void @f(double* %x, double* %y) {
  %x1 = load double, double* %x, align 8
  %yptr = bitcast double* %y to i8*
  call void @llvm.memset.p0i8.i64(i8* %yptr, i8 0, i64 8, i1 false)
  %y1 = load double, double* %y, align 8
  %x2 = fmul double %x1, %y1
  store double %x2, double* %x, align 8
  store double %x2, double* %y, align 8
  call void @llvm.memset.p0i8.i64(i8* %yptr, i8 0, i64 8, i1 false)
  ret void
}

define void @df(double* %x, double* %xp, double* %y, double* %dy) {
  tail call void @__enzyme_autodiff(i8* bitcast (void (double*, double*)* @f to i8*), double* %x, double* %xp, double* %y, double* %dy)
  ret void
}

; CHECK: define internal void @diffef(double* %x, double* %"x'", double* %y, double* %"y'")
; CHECK-NEXT: invert:
; CHECK-NEXT:   %x1 = load double, double* %x
; CHECK-NEXT:   %"yptr'ipc" = bitcast double* %"y'" to i8*
; CHECK-NEXT:   %yptr = bitcast double* %y to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* %yptr, i8 0, i64 8, i1 false)
; CHECK-NEXT:   %y1 = load double, double* %y
; CHECK-NEXT:   %x2 = fmul double %x1, %y1
; CHECK-NEXT:   store double %x2, double* %x
; CHECK-NEXT:   store double %x2, double* %y
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* %yptr, i8 0, i64 8, i1 false)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* %"yptr'ipc", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %0 = load double, double* %"y'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"y'"
; CHECK-NEXT:   %1 = fadd fast double 0.000000e+00, %0
; CHECK-NEXT:   %2 = load double, double* %"x'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'"
; CHECK-NEXT:   %3 = fadd fast double %1, %2
; CHECK-NEXT:   %[[m0diffex1:.+]] = fmul fast double %3, %y1
; CHECK-NEXT:   %[[i4:.+]] = fadd fast double 0.000000e+00, %[[m0diffex1]]
; CHECK-NEXT:   %[[m1diffey1:.+]] = fmul fast double %3, %x1
; CHECK-NEXT:   %[[i5:.+]] = fadd fast double 0.000000e+00, %[[m1diffey1]]
; CHECK-NEXT:   %[[i6:.+]] = load double, double* %"y'"
; CHECK-NEXT:   %[[i7:.+]] = fadd fast double %[[i6]], %[[i5]]
; CHECK-NEXT:   store double %[[i7]], double* %"y'"
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* %"yptr'ipc", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %[[i8:.+]] = load double, double* %"x'"
; CHECK-NEXT:   %[[i9:.+]] = fadd fast double %[[i8]], %[[i4]]
; CHECK-NEXT:   store double %[[i9]], double* %"x'"
; CHECK-NEXT:   ret void
; CHECK-NEXT: }