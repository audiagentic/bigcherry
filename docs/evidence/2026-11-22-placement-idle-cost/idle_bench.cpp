// Idle placement loop as in placement_loop: snapshot + step(idle) + sleep 2 ms; reports thread CPU and wakeups over a window.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <thread>
#include <atomic>
#include <mutex>
struct PC{unsigned seq;char pad[124];};
int main(){
  PC c{2},out{}; std::mutex mu; std::atomic<bool> stop{false}; unsigned long wakes=0; volatile unsigned sink=0;
  std::thread t([&]{
    auto t0=std::chrono::steady_clock::now();
    while(std::chrono::steady_clock::now()-t0<std::chrono::seconds(30)){
      { std::lock_guard<std::mutex> l(mu); sink+=1; }
      unsigned b=__atomic_load_n(&c.seq,__ATOMIC_ACQUIRE); std::memcpy(&out,&c,sizeof c); sink+=b+out.seq;
      std::this_thread::sleep_for(std::chrono::milliseconds(2)); ++wakes; }
    timespec ts; clock_gettime(CLOCK_THREAD_CPUTIME_ID,&ts);
    std::printf("wakes %lu (%.0f/s) thread cpu %.3f ms over 30 s = %.4f%%\n",wakes,wakes/30.0,ts.tv_sec*1e3+ts.tv_nsec/1e6,(ts.tv_sec*1e3+ts.tv_nsec/1e6)/30000.0*100);});
  t.join();
}
