; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = tail call fast double @remainder(double %x, double %y)
  ret double %0
}

define double @test_derivative1(double %x, double %y) {
entry:
  %0 = call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, metadata !"enzyme_const", double %x, double %y)
  ret double %0
}

define double @test_derivative2(double %x, double %y) {
entry:
  %0 = call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, metadata !"enzyme_const", double %y)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @remainder(double, double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %y, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = {{(fneg fast double|fsub fast double \-0\.000000e\+00,)}} %differeturn
; CHECK-NEXT:   %1 = fdiv fast double %x, %y
; CHECK-NEXT:   %2 = call fast double @llvm.round.f64(double %1)
; CHECK-NEXT:   %3 = fmul fast double %0, %2
; CHECK-NEXT:   %4 = insertvalue { double } undef, double %3, 0
; CHECK-NEXT:   ret { double } %4
; CHECK-NEXT: }



; CHECK: define internal { double } @diffetester.1(double %x, double %y, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = insertvalue { double } undef, double %differeturn, 0
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }
