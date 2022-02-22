# async
Homepage: https://github.com/d36u9/async

[[License(Boost Software License - Version 1.0)](http://www.boost.org/LICENSE_1_0.txt)]

## Welcome
async is a tiny C++ header-only high-performance library for async calls handled by a thread-pool, which is built on top of an unbounded MPMC lock-free queue.
It's written in pure C++14 (C++11 support with preprocessor macros), no dependencies on other 3rd party libraries.

Note: This library is originally designed for 64bit system. It has been tested on arch X86-64 and ARMV8(64bit), and ARMV7(32bit).

## change logs
* Jun. 2018:
  * Added support for ARMV7 & V8
  * Tested on Raspberry Pi 3 B+ with Gentoo ARMV8 64bit (Linux Pi64 4.14.44-V8 AArch64)
  * Tested on Raspberry Pi 3 B+ with Raspbian ARMV7 32bit (Linux 4.14.34-v7 armv7l)
  * Added Benchmark Results for Raspberry Pi 3 B+ ARMV8 (Linux Pi64 4.14.44-V8 AArch64)
  * Added Benchmark Results for Raspberry Pi 3 B+ ARMV7 32bit (Linux 4.14.34-v7 armv7l)
* Sept. 2017:
  * Significantly improved the performance of async::queue without bulk operations.
  * async::threadpool also benifits from this change.
  * A bounded MPMC queue `async::bounded_queue` was added to the lib, which is pretty useful for memory constrainted system or some fixed-size message pipeline design. The overall performance of this buffer based `async::bounded_queue` is comparable to bulk operations of node-based `async::queue`. `async::bounded_queue` shares the almost identical interface as `async::queue`, except for bulk operations, and a size prarameter has to be passed to `bounded_queue`'s constructor, and also added blocking methods (`blocking_enqueue` & `blocking_dequeue`). `TRAIT::NOEXCEPT_CHECK` setting is also similar to `async::queue` to help handle exceptions that may be thrown in element's ctor.  `bounded_queue` is basically a C++ implementation of [PTLQueue](https://blogs.oracle.com/dave/ptlqueue-:-a-scalable-bounded-capacity-mpmc-queue) design (Please read Dave Dice's article for details and references).

## Features
* interchangeable with std::async, accepts all kinds of callable instances, like static functions, member functions, functors, lambdas
* dynamically changeable thread-pool size at run-time
* tasks are managed in a lock-free queue
* provided lock-free queue doesn't have restricted limitation as boost::lockfree::queue
* low-latency for the task execution thanks to underlying lock-free queue

## Tested Platforms& Compilers
(old versions of OSs or compilers may work, but not tested)
* Windows 10 Visual Studio 2015+
* Linux Ubuntu 16.04 gcc4.9.2+/clang 3.8+
* MacOS Sierra 10.12.5 clang-802.0.42

## Getting Started
## Building the test& benchmark

### C++11 compilers
If your compiler only supports C++11, please edit CMakeLists.txt with the following change:
```
set(CMAKE_CXX_STANDARD 14)
#change to
set(CMAKE_CXX_STANDARD 11)
```

### Build& test with Microsoft C++ REST SDK
If your OS is Windows or has cppresetsdk installed& configured on Linux or Mac, please edit CMakeLists.txt to enable PPL test:
```
option(WITH_CPPRESTSDK "Build Cpprestsdk Test" OFF)
#to
option(WITH_CPPRESTSDK "Build Cpprestsdk Test" ON)
```


### Build for Linux or Mac (x86-64 & ARMV7&V8)
```
#to use clang (linux) with following export command
#EXPORT CC=clang-3.8
#EXPORT CXX=clang++-3.8
#run the following to set up release build, (for MasOS Xcode, you can remove -DCMAKE_BUILD_TYPE for now, and choose build type at build-time)
cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=RELEASE
#now build the release
cmake --build build --config Release
#or debug
cmake --build build --config Debug
#or other builds
cmake --build build --config RelWithDebInfo
cmake --build build --config MinSizeRel
```

### Build for Windows (X86-64)
```
#for VS 2015
cmake -H. -Bbuild -G "Visual Studio 14 2015 Win64"
#or VS 2017
cmake -H. -Bbuild -G "Visual Studio 15 2017 Win64"
#build the release from command line or you can open the project file in Visual Studio, and build from there
cmake --build build --config Release
```

## How to use it in your project/application
simply copy all headers in async sub-folder to your project, and include those headers in your source code.

## Thread Pool Indrodction
### Thread Pool intializations

```
async::threadpool tp; //by default, thread pool size will be the same number of your hardware CPU core/threads
async::threadpool tp(8); //create a thread pool with 8 threads
async::threadpool tp(0); //create a thread pool with no threads available, it's in pause mode
```

### resize the thread pool
```
async::threadpool tp(32);
...//some operations
tp.configurepool(16);// can be called at anytime (as long as tp is still valid) to reset the pool size
                     // no interurption for running tasks
```
### submit the task
*static functions, member functions, functors, lambdas are all supported
```
int foo(int i) { return ++i; }
auto pkg = tp.post(foo, i); //retuns a std::future
pkg.get(); //will block
```

## multi-producer multi-consumer unbounded lock-free queue Indrodction
The design: A simple and classic implementation. It's link-based 3-level depth nested container with local array for each level storage and simulated tagged pointer for linking.
The size of each level, and tag bits can be configured through TRAITS (please see source for details).
The queue with default traits seetings can store up to 1 Trillion elements/nodes (at least 1 Terabyte memory space).

### element type requirements
* nothrow destructible
* optional (better to be true)
  * nothrow constructible
  * nothrow move-assignable

NOTE: the exception thrown by constructor is acceptable. Although it'd be better to keep ctor noexcept if possible.
noexcept detection is turned off by default, it can be turned on by setting  `TRAIT::NOEXCEPT_CHECK` to true.
With `TRAIT::NOEXCEPT_CHECK` on(true), queue will enable exception handling if ctor or move assignment may throw exceptions.


### queue intializations
```
async::queue<T> q; //default constructor, it's unbounded

async::queue<T> q(1000); // pre-allocated 1000 storage nodes, the capcity will increase automatically after 1000 nodes are used
```
### usage
```
// enqueues a T constructed from args, supports the following constructions:
// move, if args is a T rvalue
// copy, if args is a T lvalue, or
// emplacement if args is an initializer list that can be passed to a T constructor
async::queue<T>::enqueue(Args... args)

async::queue<T>::dequeue(T& data) //type T should have move assignment operator,
//e.g.
async::queue<int> q;
q.enqueue(11);
int i(0);
q.dequeue(i);

```
### bulk operations
It's convienent for bulk data, and also can boost the throughput.
exception handling is not available in bulk operations even with `TRAIT::NOEXCEPT_CHECK` being true.
bulk operations are suitable for plain data types, like network/event messages.

```
int a[] = {1,2,3,4,5};
int b[5];
q.bulk_enqueue(std::bengin(a), 5);
auto popcount = q.bulk_dequeue(std::begin(b), 5); //popcount is the number of elemtnets sucessfully pulled from the queue.
//or like the following code:
std::vector<int> v;
auto it = std::inserter(v, std::begin(v));
popcount = q.bulk_dequeue(it, 5);
```

## Unit Test
The unit test code provides most samples for usage.

## Benchmark
NOTE: the results may vary on different OS platforms and hardware.
### thread pool benchmark
The benchmark is a simple demonstration.
NOTE: may require extra config, please see CMakeLists.txt for detailed settings
The test benchamarks the following task/job based async implementation:
* async::threadpool (this library)
* std::async
* boost::async
* AsioThreadPool (my another implementation based on boost::asio, has very stable and good performance, especially on Windows with iocp)
* Microsoft::PPL (pplx from [cpprestsdk](https://github.com/Microsoft/cpprestsdk) on Linux& MacOS or PPL on windows)


e.g. Windows 10 64bit Intel i7-6700K 16GB RAM 480GB SSD Visual Studio 2017 (cl 19.11.25507.1 x64)
```
Benchmark Test Run: 1 Producers 7(* not applied) Consumers  with 21000 tasks and run 100 batches
  async::threapool (time/task) avg: 1130 ns  max: 1227 ns  min: 1066 ns avg_task_post: 1032 ns
       *std::async (time/task) avg: 1469 ns  max: 1549 ns  min: 1423 ns avg_task_post: 1250 ns
   *Microsoft::PPL (time/task) avg: 1148 ns  max: 1216 ns  min: 1114 ns avg_task_post: 1088 ns
    AsioThreadPool (time/task) avg: 1166 ns  max: 1319 ns  min: 1013 ns avg_task_post: 1073 ns
     *boost::async (time/task) avg: 29153 ns  max: 30028 ns  min: 27990 ns avg_task_post: 23343 ns
...
Benchmark Test Run: 4 Producers 4(* not applied) Consumers  with 21000 tasks and run 100 batches
  async::threapool (time/task) avg: 439 ns  max: 557 ns  min: 398 ns avg_task_post: 356 ns
       *std::async (time/task) avg: 800 ns  max: 890 ns  min: 759 ns avg_task_post: 629 ns
   *Microsoft::PPL (time/task) avg: 666 ns  max: 701 ns  min: 640 ns avg_task_post: 605 ns
    AsioThreadPool (time/task) avg: 448 ns  max: 541 ns  min: 389 ns avg_task_post: 365 ns
     *boost::async (time/task) avg: 32419 ns  max: 33296 ns  min: 30105 ns avg_task_post: 25561 ns
...
Benchmark Test Run: 7 Producers 1(* not applied) Consumers  with 21000 tasks and run 100 batches
  async::threapool (time/task) avg: 262 ns  max: 300 ns  min: 252 ns avg_task_post: 176 ns
       *std::async (time/task) avg: 873 ns  max: 961 ns  min: 821 ns avg_task_post: 701 ns
   *Microsoft::PPL (time/task) avg: 727 ns  max: 755 ns  min: 637 ns avg_task_post: 662 ns
    AsioThreadPool (time/task) avg: 607 ns  max: 645 ns  min: 567 ns avg_task_post: 210 ns
     *boost::async (time/task) avg: 33158 ns  max: 150331 ns  min: 28560 ns avg_task_post: 28655 ns
```

e.g. Ubuntu 17.04 Intel i7-6700K 16GB RAM 100GB HDD gcc 6.3.0
```
Benchmark Test Run: 1 Producers 7(* not applied) Consumers  with 21000 tasks and run 100 batches
  async::threapool (time/task) avg: 1320 ns  max: 1357 ns  min: 1301 ns avg_task_post: 1266 ns
       *std::async (time/task) avg: 11817 ns  max: 12469 ns  min: 11533 ns avg_task_post: 9580 ns
   *Microsoft::PPL (time/task) avg: 1368 ns  max: 1498 ns  min: 1325 ns avg_task_post: 1349 ns
    AsioThreadPool (time/task) avg: 1475 ns  max: 1499 ns  min: 1318 ns avg_task_post: 1332 ns
     *boost::async (time/task) avg: 4574 ns  max: 4697 ns  min: 4450 ns avg_task_post: 4531 ns
...
Benchmark Test Run: 4 Producers 4(* not applied) Consumers  with 21000 tasks and run 100 batches
  async::threapool (time/task) avg: 516 ns  max: 688 ns  min: 239 ns avg_task_post: 522 ns
       *std::async (time/task) avg: 41630 ns  max: 44316 ns  min: 41334 ns avg_task_post: 38151 ns
   *Microsoft::PPL (time/task) avg: 3652 ns  max: 3710 ns  min: 3598 ns avg_task_post: 3629 ns
    AsioThreadPool (time/task) avg: 529 ns  max: 814 ns  min: 494 ns avg_task_post: 447 ns
     *boost::async (time/task) avg: 14634 ns  max: 14669 ns  min: 14598 ns avg_task_post: 14583 ns
...
Benchmark Test Run: 7 Producers 1(* not applied) Consumers  with 21000 tasks and run 100 batches
  async::threapool (time/task) avg: 398 ns  max: 468 ns  min: 337 ns avg_task_post: 177 ns
       *std::async (time/task) avg: 44603 ns  max: 46904 ns  min: 44272 ns avg_task_post: 40877 ns
   *Microsoft::PPL (time/task) avg: 3714 ns  max: 3816 ns  min: 3656 ns avg_task_post: 3690 ns
    AsioThreadPool (time/task) avg: 564 ns  max: 605 ns  min: 533 ns avg_task_post: 253 ns
     *boost::async (time/task) avg: 20421 ns  max: 21738 ns  min: 19105 ns avg_task_post: 20375 ns
```

e.g. MacOS 10.12.5 clang Intel i7-6700K 16GB RAM 250GB SSD clang-802.0.42 (Microsoft::PPL(cpprestsdk::pplx) is superisingly good compared with other libraries on MacOS, not sure if it's due to some comipiler optimization)
```
Benchmark Test Run: 1 Producers 7(* not applied) Consumers  with 21000 tasks and run 100 batches
  async::threapool (time/task) avg: 8517 ns  max: 8641 ns  min: 7400 ns avg_task_post: 8393 ns
       *std::async (time/task) avg: 13618 ns  max: 13845 ns  min: 13276 ns avg_task_post: 13476 ns
   *Microsoft::PPL (time/task) avg: 747 ns  max: 938 ns  min: 626 ns avg_task_post: 718 ns
    AsioThreadPool (time/task) avg: 8647 ns  max: 8807 ns  min: 8558 ns avg_task_post: 8524 ns
     *boost::async (time/task) avg: 11732 ns  max: 12028 ns  min: 11526 ns avg_task_post: 11698 ns
...
Benchmark Test Run: 4 Producers 4(* not applied) Consumers  with 21000 tasks and run 100 batches
  async::threapool (time/task) avg: 5964 ns  max: 6017 ns  min: 5790 ns avg_task_post: 5830 ns
       *std::async (time/task) avg: 9690 ns  max: 10043 ns  min: 9132 ns avg_task_post: 9531 ns
   *Microsoft::PPL (time/task) avg: 380 ns  max: 425 ns  min: 342 ns avg_task_post: 353 ns
    AsioThreadPool (time/task) avg: 6173 ns  max: 6459 ns  min: 6116 ns avg_task_post: 6042 ns
     *boost::async (time/task) avg: 8643 ns  max: 9470 ns  min: 8513 ns avg_task_post: 8591 ns
...
Benchmark Test Run: 7 Producers 1(* not applied) Consumers  with 21000 tasks and run 100 batches
  async::threapool (time/task) avg: 3469 ns  max: 3527 ns  min: 3415 ns avg_task_post: 3339 ns
       *std::async (time/task) avg: 10902 ns  max: 11164 ns  min: 10709 ns avg_task_post: 10738 ns
   *Microsoft::PPL (time/task) avg: 367 ns  max: 426 ns  min: 326 ns avg_task_post: 323 ns
    AsioThreadPool (time/task) avg: 3920 ns  max: 3975 ns  min: 3832 ns avg_task_post: 3409 ns
     *boost::async (time/task) avg: 9800 ns  max: 10223 ns  min: 9196 ns avg_task_post: 9744 ns
```

e.g. Windows 7 64bit Intel i7-4790 16GB RAM Visual Studio 2015 Update 3
```
Benchmark Test Run: 1 Producers 7(* not applied) Consumers  with 21000 tasks and run 100 batches
  async::threapool (time/task) avg: 809 ns  max: 924 ns  min: 687 ns avg_task_post: 774 ns
       *std::async (time/task) avg: 1914 ns  max: 2032 ns  min: 1790 ns avg_task_post: 1877 ns
   *Microsoft::PPL (time/task) avg: 1718 ns  max: 2181 ns  min: 1623 ns avg_task_post: 1677 ns
    AsioThreadPool (time/task) avg: 1100 ns  max: 1137 ns  min: 1076 ns avg_task_post: 1065 ns
     *boost::async (time/task) avg: 191532 ns  max: 203716 ns  min: 186114 ns avg_task_post: 191507 ns
...
Benchmark Test Run: 4 Producers 4(* not applied) Consumers  with 21000 tasks and run 100 batches
  async::threapool (time/task) avg: 423 ns  max: 538 ns  min: 338 ns avg_task_post: 388 ns
       *std::async (time/task) avg: 1249 ns  max: 1279 ns  min: 1233 ns avg_task_post: 1211 ns
   *Microsoft::PPL (time/task) avg: 1229 ns  max: 1246 ns  min: 1208 ns avg_task_post: 1186 ns
    AsioThreadPool (time/task) avg: 563 ns  max: 577 ns  min: 499 ns avg_task_post: 528 ns
     *boost::async (time/task) avg: 95484 ns  max: 112569 ns  min: 93808 ns avg_task_post: 95458 ns
...
Benchmark Test Run: 7 Producers 1(* not applied) Consumers  with 21000 tasks and run 100 batches
  async::threapool (time/task) avg: 267 ns  max: 323 ns  min: 255 ns avg_task_post: 232 ns
       *std::async (time/task) avg: 1202 ns  max: 1257 ns  min: 1182 ns avg_task_post: 1009 ns
   *Microsoft::PPL (time/task) avg: 1199 ns  max: 1262 ns  min: 1175 ns avg_task_post: 988 ns
    AsioThreadPool (time/task) avg: 783 ns  max: 960 ns  min: 706 ns avg_task_post: 375 ns
     *boost::async (time/task) avg: 103572 ns  max: 107041 ns  min: 101993 ns avg_task_post: 103542 ns
```

e.g. Gentoo ARMV8 64bit (Linux Pi64 4.14.44-V8 AArch64) gcc 7.3.0 on Raspberry Pi 3 B+
```
Benchmark Test Run: 1 Producers 3(* not applied) Consumers  with 21000 tasks and run 100 batches
  async::threapool (time/task) avg: 7809 ns  max: 10467 ns  min: 7453 ns avg_task_post: 7261 ns
       *std::async (time/task) avg: 139664 ns  max: 3453077 ns  min: 104589 ns avg_task_post: 117819 ns
    AsioThreadPool (time/task) avg: 6545 ns  max: 8804 ns  min: 5678 ns avg_task_post: 5654 ns
     *boost::async (time/task) avg: 37629 ns  max: 38978 ns  min: 36769 ns avg_task_post: 36933 ns

Benchmark Test Run: 2 Producers 2(* not applied) Consumers  with 21000 tasks and run 100 batches
  async::threapool (time/task) avg: 2207 ns  max: 4084 ns  min: 1809 ns avg_task_post: 1325 ns
       *std::async (time/task) avg: 431781 ns  max: 17500817 ns  min: 91919 ns avg_task_post: 407595 ns
    AsioThreadPool (time/task) avg: 2251 ns  max: 3351 ns  min: 1839 ns avg_task_post: 1405 ns
     *boost::async (time/task) avg: 48456 ns  max: 50578 ns  min: 46698 ns avg_task_post: 47753 ns

Benchmark Test Run: 3 Producers 1(* not applied) Consumers  with 21000 tasks and run 100 batches
  async::threapool (time/task) avg: 3346 ns  max: 3974 ns  min: 2635 ns avg_task_post: 1017 ns
       *std::async (time/task) avg: 110853 ns  max: 768224 ns  min: 103045 ns avg_task_post: 86361 ns
    AsioThreadPool (time/task) avg: 3828 ns  max: 4209 ns  min: 3354 ns avg_task_post: 976 ns
     *boost::async (time/task) avg: 59094 ns  max: 67042 ns  min: 54802 ns avg_task_post: 58365 ns
```

### queue benchmark
The benchmark uses producers-consumers model, and doesn't provide all the detailed measurements.
* async::bounded_queue
* async::queue
* boost::lockfree::queue
* boost::lockfree::spsc_queue  (only for single-producer-single-consumer test)

e.g. Windows 10 64bit Intel i7-6700K 16GB RAM 480GB SSD Visual Studio 2017 (cl 19.11.25507.1 x64)
```
Single Producer Single Consumer Benchmark with 10000 Ops and run 1000 batches
Benchmark Test Run: 1 Producers 1 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 18 ns  max: 55 ns  min: 17 ns
async::queue::bulk(16) (time/op) avg: 26 ns  max: 50 ns  min: 23 ns
          async::queue (time/op) avg: 28 ns  max: 66 ns  min: 27 ns
boost::lockfree::queue (time/op) avg: 167 ns  max: 195 ns  min: 70 ns
boost::lockfree::spsc_queue (time/op) avg: 10 ns  max: 38 ns  min: 8 ns

Benchmark Test Run: 1 Producers 7 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 27 ns  max: 62 ns  min: 25 ns
async::queue::bulk(16) (time/op) avg: 28 ns  max: 124 ns  min: 24 ns
          async::queue (time/op) avg: 42 ns  max: 115 ns  min: 29 ns
boost::lockfree::queue (time/op) avg: 240 ns  max: 576 ns  min: 119 ns

Benchmark Test Run: 2 Producers 6 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 44 ns  max: 78 ns  min: 29 ns
async::queue::bulk(16) (time/op) avg: 34 ns  max: 109 ns  min: 28 ns
          async::queue (time/op) avg: 90 ns  max: 122 ns  min: 44 ns
boost::lockfree::queue (time/op) avg: 213 ns  max: 227 ns  min: 161 ns

Benchmark Test Run: 3 Producers 5 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 53 ns  max: 82 ns  min: 27 ns
async::queue::bulk(16) (time/op) avg: 34 ns  max: 107 ns  min: 29 ns
          async::queue (time/op) avg: 100 ns  max: 114 ns  min: 51 ns
boost::lockfree::queue (time/op) avg: 197 ns  max: 207 ns  min: 186 ns

Benchmark Test Run: 4 Producers 4 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 31 ns  max: 81 ns  min: 25 ns
async::queue::bulk(16) (time/op) avg: 31 ns  max: 104 ns  min: 28 ns
          async::queue (time/op) avg: 93 ns  max: 117 ns  min: 73 ns
boost::lockfree::queue (time/op) avg: 211 ns  max: 222 ns  min: 162 ns

Benchmark Test Run: 5 Producers 3 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 52 ns  max: 79 ns  min: 30 ns
async::queue::bulk(16) (time/op) avg: 33 ns  max: 103 ns  min: 29 ns
          async::queue (time/op) avg: 94 ns  max: 126 ns  min: 74 ns
boost::lockfree::queue (time/op) avg: 199 ns  max: 217 ns  min: 174 ns

Benchmark Test Run: 6 Producers 2 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 49 ns  max: 81 ns  min: 35 ns
async::queue::bulk(16) (time/op) avg: 33 ns  max: 60 ns  min: 28 ns
          async::queue (time/op) avg: 97 ns  max: 134 ns  min: 51 ns
boost::lockfree::queue (time/op) avg: 185 ns  max: 198 ns  min: 152 ns

Benchmark Test Run: 7 Producers 1 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 36 ns  max: 81 ns  min: 34 ns
async::queue::bulk(16) (time/op) avg: 30 ns  max: 60 ns  min: 26 ns
          async::queue (time/op) avg: 48 ns  max: 89 ns  min: 45 ns
boost::lockfree::queue (time/op) avg: 161 ns  max: 179 ns  min: 120 ns
```

e.g. MacOS 10.12.5 Intel i7-6700K 16GB RAM 250GB SSD clang-802.0.42
```
SSingle Producer Single Consumer Benchmark with 10000 Ops and run 1000 batches
Benchmark Test Run: 1 Producers 1 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 12 ns  max: 37 ns  min: 12 ns
async::queue::bulk(16) (time/op) avg: 26 ns  max: 54 ns  min: 25 ns
          async::queue (time/op) avg: 23 ns  max: 61 ns  min: 23 ns
boost::lockfree::queue (time/op) avg: 156 ns  max: 172 ns  min: 118 ns
boost::lockfree::spsc_queue (time/op) avg: 11 ns  max: 30 ns  min: 5 ns

Benchmark Test Run: 1 Producers 7 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 84 ns  max: 98 ns  min: 60 ns
async::queue::bulk(16) (time/op) avg: 27 ns  max: 125 ns  min: 24 ns
          async::queue (time/op) avg: 104 ns  max: 115 ns  min: 92 ns
boost::lockfree::queue (time/op) avg: 231 ns  max: 326 ns  min: 213 ns

Benchmark Test Run: 2 Producers 6 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 82 ns  max: 100 ns  min: 61 ns
async::queue::bulk(16) (time/op) avg: 36 ns  max: 108 ns  min: 31 ns
          async::queue (time/op) avg: 102 ns  max: 122 ns  min: 90 ns
boost::lockfree::queue (time/op) avg: 192 ns  max: 229 ns  min: 184 ns

Benchmark Test Run: 3 Producers 5 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 79 ns  max: 93 ns  min: 61 ns
async::queue::bulk(16) (time/op) avg: 31 ns  max: 94 ns  min: 29 ns
          async::queue (time/op) avg: 98 ns  max: 116 ns  min: 70 ns
boost::lockfree::queue (time/op) avg: 189 ns  max: 198 ns  min: 175 ns

Benchmark Test Run: 4 Producers 4 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 77 ns  max: 146 ns  min: 56 ns
async::queue::bulk(16) (time/op) avg: 28 ns  max: 92 ns  min: 26 ns
          async::queue (time/op) avg: 93 ns  max: 167 ns  min: 73 ns
boost::lockfree::queue (time/op) avg: 200 ns  max: 218 ns  min: 182 ns

Benchmark Test Run: 5 Producers 3 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 76 ns  max: 92 ns  min: 48 ns
async::queue::bulk(16) (time/op) avg: 27 ns  max: 89 ns  min: 24 ns
          async::queue (time/op) avg: 97 ns  max: 140 ns  min: 83 ns
boost::lockfree::queue (time/op) avg: 200 ns  max: 211 ns  min: 163 ns

Benchmark Test Run: 6 Producers 2 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 80 ns  max: 98 ns  min: 59 ns
async::queue::bulk(16) (time/op) avg: 28 ns  max: 97 ns  min: 24 ns
          async::queue (time/op) avg: 105 ns  max: 122 ns  min: 78 ns
boost::lockfree::queue (time/op) avg: 182 ns  max: 194 ns  min: 153 ns

Benchmark Test Run: 7 Producers 1 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 86 ns  max: 103 ns  min: 64 ns
async::queue::bulk(16) (time/op) avg: 27 ns  max: 82 ns  min: 23 ns
          async::queue (time/op) avg: 107 ns  max: 127 ns  min: 91 ns
boost::lockfree::queue (time/op) avg: 154 ns  max: 180 ns  min: 146 ns
```

e.g. Ubuntu 17.04 Intel i7-6700K 16GB RAM 100GB HDD gcc 6.3.0
```
Single Producer Single Consumer Benchmark with 10000 Ops and run 1000 batches
Benchmark Test Run: 1 Producers 1 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 12 ns  max: 71 ns  min: 11 ns
async::queue::bulk(16) (time/op) avg: 65 ns  max: 134 ns  min: 24 ns
          async::queue (time/op) avg: 48 ns  max: 107 ns  min: 33 ns
boost::lockfree::queue (time/op) avg: 179 ns  max: 198 ns  min: 60 ns
boost::lockfree::spsc_queue (time/op) avg: 7 ns  max: 47 ns  min: 4 ns

Benchmark Test Run: 1 Producers 7 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 68 ns  max: 505 ns  min: 35 ns
async::queue::bulk(16) (time/op) avg: 29 ns  max: 135 ns  min: 25 ns
          async::queue (time/op) avg: 93 ns  max: 138 ns  min: 73 ns
boost::lockfree::queue (time/op) avg: 234 ns  max: 292 ns  min: 208 ns

Benchmark Test Run: 2 Producers 6 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 68 ns  max: 106 ns  min: 39 ns
async::queue::bulk(16) (time/op) avg: 35 ns  max: 117 ns  min: 19 ns
          async::queue (time/op) avg: 92 ns  max: 135 ns  min: 79 ns
boost::lockfree::queue (time/op) avg: 193 ns  max: 227 ns  min: 175 ns

Benchmark Test Run: 3 Producers 5 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 73 ns  max: 251 ns  min: 49 ns
async::queue::bulk(16) (time/op) avg: 31 ns  max: 110 ns  min: 26 ns
          async::queue (time/op) avg: 96 ns  max: 178 ns  min: 70 ns
boost::lockfree::queue (time/op) avg: 179 ns  max: 359 ns  min: 164 ns

Benchmark Test Run: 4 Producers 4 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 81 ns  max: 220 ns  min: 61 ns
async::queue::bulk(16) (time/op) avg: 27 ns  max: 114 ns  min: 23 ns
          async::queue (time/op) avg: 102 ns  max: 159 ns  min: 74 ns
boost::lockfree::queue (time/op) avg: 177 ns  max: 541 ns  min: 162 ns

Benchmark Test Run: 5 Producers 3 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 83 ns  max: 443 ns  min: 53 ns
async::queue::bulk(16) (time/op) avg: 26 ns  max: 297 ns  min: 23 ns
          async::queue (time/op) avg: 110 ns  max: 512 ns  min: 79 ns
boost::lockfree::queue (time/op) avg: 176 ns  max: 505 ns  min: 161 ns

Benchmark Test Run: 6 Producers 2 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 83 ns  max: 437 ns  min: 36 ns
async::queue::bulk(16) (time/op) avg: 26 ns  max: 261 ns  min: 23 ns
          async::queue (time/op) avg: 112 ns  max: 449 ns  min: 84 ns
boost::lockfree::queue (time/op) avg: 178 ns  max: 547 ns  min: 164 ns

Benchmark Test Run: 7 Producers 1 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 90 ns  max: 805 ns  min: 28 ns
async::queue::bulk(16) (time/op) avg: 26 ns  max: 78 ns  min: 21 ns
          async::queue (time/op) avg: 123 ns  max: 695 ns  min: 80 ns
boost::lockfree::queue (time/op) avg: 195 ns  max: 615 ns  min: 154 ns
```

e.g. Gentoo ARMV8 64bit (Linux Pi64 4.14.44-V8 AArch64) gcc 7.3.0 on Raspberry Pi 3 B+
```
Single Producer Single Consumer Benchmark with 10000 Ops and run 1000 batches
Benchmark Test Run: 1 Producers 1 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 67 ns  max: 697 ns  min: 53 ns
async::queue::bulk(16) (time/op) avg: 144 ns  max: 434 ns  min: 130 ns
          async::queue (time/op) avg: 141 ns  max: 441 ns  min: 115 ns
boost::lockfree::queue (time/op) avg: 182 ns  max: 514 ns  min: 168 ns
boost::lockfree::spsc_queue (time/op) avg: 62 ns  max: 430 ns  min: 53 ns

Benchmark Test Run: 1 Producers 3 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 72 ns  max: 574 ns  min: 59 ns
async::queue::bulk(16) (time/op) avg: 141 ns  max: 515 ns  min: 116 ns
          async::queue (time/op) avg: 181 ns  max: 590 ns  min: 134 ns
boost::lockfree::queue (time/op) avg: 192 ns  max: 1045 ns  min: 172 ns

Benchmark Test Run: 2 Producers 2 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 82 ns  max: 457 ns  min: 65 ns
async::queue::bulk(16) (time/op) avg: 99 ns  max: 701 ns  min: 84 ns
          async::queue (time/op) avg: 124 ns  max: 550 ns  min: 108 ns
boost::lockfree::queue (time/op) avg: 151 ns  max: 847 ns  min: 138 ns

Benchmark Test Run: 3 Producers 1 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 88 ns  max: 538 ns  min: 67 ns
async::queue::bulk(16) (time/op) avg: 89 ns  max: 717 ns  min: 71 ns
          async::queue (time/op) avg: 131 ns  max: 631 ns  min: 118 ns
boost::lockfree::queue (time/op) avg: 165 ns  max: 644 ns  min: 149 ns
```

e.g. Raspbian ARMV7 32bit (Linux 4.14.34-v7 armv7l) gcc 6.3.0 on Raspberry Pi 3 B+
```
Single Producer Single Consumer Benchmark with 10000 Ops and run 1000 batches
Benchmark Test Run: 1 Producers 1 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 227 ns  max: 912 ns  min: 179 ns
async::queue::bulk(16) (time/op) avg: 442 ns  max: 1236 ns  min: 365 ns
          async::queue (time/op) avg: 423 ns  max: 1249 ns  min: 364 ns
boost::lockfree::queue (time/op) avg: 474 ns  max: 1017 ns  min: 410 ns
boost::lockfree::spsc_queue (time/op) avg: 70 ns  max: 761 ns  min: 48 ns

Benchmark Test Run: 1 Producers 3 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 241 ns  max: 1482 ns  min: 187 ns
async::queue::bulk(16) (time/op) avg: 470 ns  max: 1259 ns  min: 354 ns
          async::queue (time/op) avg: 488 ns  max: 1482 ns  min: 375 ns
boost::lockfree::queue (time/op) avg: 462 ns  max: 1158 ns  min: 427 ns


Benchmark Test Run: 2 Producers 2 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 208 ns  max: 348 ns  min: 158 ns
async::queue::bulk(16) (time/op) avg: 285 ns  max: 543 ns  min: 237 ns
          async::queue (time/op) avg: 306 ns  max: 761 ns  min: 234 ns
boost::lockfree::queue (time/op) avg: 334 ns  max: 1481 ns  min: 261 ns


Benchmark Test Run: 3 Producers 1 Consumers  with 10000 Ops and run 1000 batches
  async::bounded_queue (time/op) avg: 241 ns  max: 884 ns  min: 192 ns
async::queue::bulk(16) (time/op) avg: 210 ns  max: 651 ns  min: 180 ns
          async::queue (time/op) avg: 439 ns  max: 682 ns  min: 375 ns
boost::lockfree::queue (time/op) avg: 420 ns  max: 903 ns  min: 320 ns
```

## coding style
all code has been formated by clang-format. It may be more easy to read in text editor or may be not :)

## Many Thanks to 3rd party and their developers
* [Boost](http://www.boost.org/)
* [Boost CMake](https://github.com/Orphis/boost-cmake) Easy Boost integration in CMake projects!
* [Catch](https://github.com/philsquared/Catch) A powerful test framework for unit test.
* [cpprestsdk](https://github.com/Microsoft/cpprestsdk) The C++ REST SDK is a Microsoft project for cloud-based client-server communication in native code using a modern asynchronous C++ API design.
* [rlutil](https://github.com/tapio/rlutil) provides cross-platform console-mode functions to position and colorize text.
* [sakaki](https://github.com/sakaki-/gentoo-on-rpi3-64bit) Bootable 64-bit Gentoo image for the Raspberry Pi 3 B / B+, with Linux 4.14
