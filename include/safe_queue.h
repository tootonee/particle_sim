//
// Created by Admin on 14.03.2024.
//

#ifndef INTEGRATE_SERIAL_SAFE_QUEUE_H
#define INTEGRATE_SERIAL_SAFE_QUEUE_H

#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>

template<typename T>
class ThreadSafeQueue {

private:
    std::queue<T> data_queue;
    mutable std::mutex mx;
    std::condition_variable cv;

public:
    ThreadSafeQueue() = default;
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator = (
            const ThreadSafeQueue&) = delete;

    void push(T new_value){
        std::lock_guard<std::mutex> lck(mx);
        data_queue.push(new_value);
        cv.notify_one();
    }

    std::shared_ptr<T> try_pop(){
        std::lock_guard<std::mutex> lck(mx);
        if( data_queue.empty() ) return std::shared_ptr<T>();
        std::shared_ptr<T> res(std::make_shared<T>( data_queue.front() ));
        data_queue.pop();
        return res;
    }

    bool try_pop(T& value)
    {
        std::lock_guard<std::mutex> lk(mx);
        if(data_queue.empty())
            return false;
        value=data_queue.front();
        data_queue.pop();
        return true;
    }

    std::shared_ptr<T> wait_and_pop(){
        std::unique_lock<std::mutex> lck(mx);
        cv.wait(lck, [ this ] {
            return !data_queue.empty();
        });
        std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
        data_queue.pop();
        return res;
    }

    bool empty() const{
        std::lock_guard<std::mutex> lck(mx);
        return data_queue.empty();
    }
};

#endif //INTEGRATE_SERIAL_SAFE_QUEUE_H