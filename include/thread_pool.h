//
// Created by Admin on 28.03.2024.
//

#ifndef INTEGRATE_PARALLEL_QUEUE_THREAD_POOL_H
#define INTEGRATE_PARALLEL_QUEUE_THREAD_POOL_H

#include <vector>
#include <functional>
#include "safe_queue.h"
#include <thread>
#include <atomic>
#include <future>


class join_threads {
    std::vector<std::thread>& threads;

public:
    explicit join_threads(std::vector<std::thread>& threads_): threads(threads_) {}

    ~join_threads() {
        for(auto& thread : threads) {
            if(thread.joinable()) {
                thread.join();
            }
        }
    }
};

class thread_pool
{
    std::atomic_bool done;
    ThreadSafeQueue<std::function<void()>> work_queue;
    std::vector<std::thread> threads;
    join_threads joiner;
    void worker_thread()
    {
        while(!done)
        {
            std::function<void()> task;
            if(work_queue.try_pop(task))
            {
                task();
            }
            else
            {
                std::this_thread::yield();
            }
        }
    }
public:
    thread_pool():
            done(false),joiner(threads)
    {
        unsigned const thread_count=std::thread::hardware_concurrency();
        try
        {
            for(unsigned i=0;i<thread_count;++i)
            {
                threads.push_back(
                        std::thread(&thread_pool::worker_thread,this));
            }
        }
        catch(...)
        {
            done=true;
            throw;
        }
    }
    ~thread_pool()
    {
        done=true;
    }

    template<typename FunctionType>
    auto submit(FunctionType f) -> std::future<decltype(f())> {
        using result_type = decltype(f());

        auto task_ptr = std::make_shared<std::packaged_task<result_type()>>(
                std::function<result_type()>(f)
        );

        std::future<result_type> result = task_ptr->get_future();
        work_queue.push([task_ptr]() { (*task_ptr)(); });

        return result;
    }
};

#endif //INTEGRATE_PARALLEL_QUEUE_THREAD_POOL_H