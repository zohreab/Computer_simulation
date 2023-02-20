"""Companion code to https://realpython.com/simulation-with-simpy/

'Simulating Real-World Processes With SimPy'

Python version: 3.7.3
SimPy version: 3.0.11
"""
import threading
from time import sleep

import simpy
import numpy as np
from threading import Lock

priority_interval = 10000
X = 0
flag = True
K = 5
check_arrival_time = 5
T1 = 1
T2 = 2
Jobs = []
Q_second_layer_lens_sum = [0, 0, 0]
q_first_layer_sum = 0
x0 = 0
x1 = 0
x2 = 0
x3 = 0
lock = Lock()
t1 = 0
t2 = 0
cpu_working_time = 0
number_of_transformed_j = 0
next_line = 1


class Job:
    def __init__(self, run_time, arrival_time, id):
        self.run_time = run_time
        self.run_time1 = run_time
        self.arrival_time = arrival_time
        self.finish_time = 0
        self.lane = 0
        self.id = id
        self.waiting_time = [0, 0, 0, 0]
        self.priority = 0
        self.starvation = False
        self.starvation_time = 0
        self.incorrect_lane = False


class CPUScheduler(object):
    def __init__(self, env):
        self.env = env
        self.cpu = simpy.PriorityResource(env)
        self.waiting = simpy.PriorityResource(env)

    def change_vars(self, time):
        global cpu_working_time
        Q_second_layer_lens_sum[0] += x1 * time
        Q_second_layer_lens_sum[1] += x2 * time
        Q_second_layer_lens_sum[2] += x3 * time
        cpu_working_time += time

    def check_starvation(self, job):
        global x1, x2, x3

        if self.env.now - (job.arrival_time + (job.run_time1-job.run_time)) > job.starvation_time:
            job.starvation = True
            job.run_time = 0
            if job.lane == 1:
                job.waiting_time[1] = self.env.now - (job.arrival_time + job.waiting_time[0])
                x1 -= 1
            elif job.lane == 2:
                job.waiting_time[2] = self.env.now - (job.arrival_time + job.waiting_time[0] + job.waiting_time[1] + T1)
                x2 -= 1
            elif job.lane == 3:
                job.waiting_time[3] = self.env.now - (
                        job.arrival_time + job.waiting_time[0] + job.waiting_time[1] + job.waiting_time[2] + T1 + T2)
                x3 -= 1


    def decide_next_line(self):
        global x1, x2, x3
        if x1 != 0 and x2 != 0 and x3 != 0:
            return np.random.choice([1, 2, 3], p=[0.8, 0.1, 0.1])
        elif x1 != 0 and x2 != 0 and x3 == 0:
            return np.random.choice([1, 2], p=[0.89, 0.11])
        elif x1 != 0 and x2 == 0 and x3 != 0:
            return np.random.choice([1, 3], p=[0.89, 0.11])
        elif x1 == 0 and x2 != 0 and x3 != 0:
            return np.random.choice([2, 3], p=[0.5, 0.5])
        elif x1 != 0 and x2 == 0 and x3 == 0:
            return 1
        elif x1 == 0 and x2 != 0 and x3 == 0:
            return 2
        elif x1 == 0 and x2 == 0 and x3 != 0:
            return 3
        else:
            return 1

    def do_job(self, job):
        global X, x1, x2, x3, number_of_transformed_j, flag, next_line
        number_of_transformed_j = 0
        flag = True
        if next_line != job.lane:
            job.incorrect_lane = True
            return
        self.check_starvation(job)
        if job.starvation:
            yield self.env.timeout(0)
            next_line = self.decide_next_line()
            return
        if job.lane == 1:
            if job.run_time <= T1:
                job.waiting_time[1] = self.env.now - (job.arrival_time + job.waiting_time[0])
                yield self.env.timeout(job.run_time)
                job.finish_time = self.env.now
                x1 -= 1
                self.change_vars(job.run_time)
                job.run_time = 0
            else:
                job.lane = 2
                job.waiting_time[1] = self.env.now - (job.arrival_time + job.waiting_time[0])
                yield self.env.timeout(T1)
                x1 -= 1
                self.change_vars(T1)
                x2 += 1
                job.run_time -= T1
        elif job.lane == 2:
            if job.run_time <= T2:
                job.waiting_time[2] = self.env.now - (job.arrival_time + job.waiting_time[0] + job.waiting_time[1] + T1)
                yield self.env.timeout(job.run_time)
                job.finish_time = self.env.now
                x2 -= 1
                self.change_vars(job.run_time)
                job.run_time = 0
            else:
                job.waiting_time[2] = self.env.now - (job.arrival_time + job.waiting_time[0] + job.waiting_time[1] + T1)
                job.lane = 3
                yield self.env.timeout(T2)
                job.run_time -= T2
                x2 -= 1
                self.change_vars(T2)
                x3 += 1
        elif job.lane == 3:
            job.lane = 4
            job.waiting_time[3] = self.env.now - (
                    job.arrival_time + job.waiting_time[0] + job.waiting_time[1] + job.waiting_time[2] + T1 + T2)
            yield self.env.timeout(job.run_time)
            job.finish_time = self.env.now
            x3 -= 1
            self.change_vars(job.run_time)
            job.run_time = 0

        next_line = self.decide_next_line()

    def correct_lane(self, job):
        global X, x0, x1, t1, t2, q_first_layer_sum, number_of_transformed_j, flag, next_line
        if X >= K and flag:
            pass
        else:
            flag = False
            X += 1
            x1 += 1
            next_line = self.decide_next_line()
            job.waiting_time[0] = self.env.now - job.arrival_time
            lock.acquire()
            t2 = self.env.now
            q_first_layer_sum += x0 * (t2 - t1)
            t1 = t2
            x0 -= 1
            lock.release()
            number_of_transformed_j += 1
            job.lane = 1
            if number_of_transformed_j == K:
                flag = True
                number_of_transformed_j = 0
        yield self.env.timeout(0)




def schedule_job(env, job, cpu_scheduler):
    global X, priority_interval

    while job.lane == 0:

        yield env.timeout((check_arrival_time - (env.now % check_arrival_time)))
        with cpu_scheduler.cpu.request(priority=1 + job.arrival_time + job.priority * priority_interval) as request2:
            yield request2
            yield env.process(cpu_scheduler.correct_lane(job))



    while job.run_time != 0:

        if job.incorrect_lane:
            yield env.timeout(0.5)
            job.incorrect_lane = False
        with cpu_scheduler.cpu.request(priority=((job.arrival_time) + ((job.lane + 4) * priority_interval))) as request:
            yield request
            yield env.process(cpu_scheduler.do_job(job))



    X -= 1


def run_scheduler(env, poi, Y, Z):
    global x0, q_first_layer_sum, t1, t2
    cpu_scheduler = CPUScheduler(env)
    np.random.seed(42)
    while True:
        yield env.timeout(int(np.random.exponential(poi)) + 1)
        job = Job(int(np.random.exponential(Y)) + 1, env.now, len(Jobs))
        job.priority = np.random.choice([3, 2, 1], p=[0.7, 0.2, 0.1])
        Jobs.append(job)
        job.lane = 0
        job.starvation_time = np.random.exponential(Z)
        lock.acquire()
        t2 = env.now
        q_first_layer_sum += x0 * (t2 - t1)
        t1 = t2
        x0 += 1
        lock.release()
        env.process(schedule_job(env, job, cpu_scheduler))


def get_input():
    print("Enter poisson distribution parameter (for inter arrival time): ")
    poi = float(input())
    print("Enter exponential distribution parameter (service time): ")
    Y = int(input())
    print("Enter exponential distribution parameter (expire time): ")
    Z = int(input())
    print("Enter simulation time: ")
    end_time = int(input())
    return int(1/poi), Y, Z, end_time


def main():
    # Setup

    # Run the simulation
    poi, Y, Z, end_time = get_input()
    env = simpy.Environment()
    env.process(run_scheduler(env, poi, Y, Z))
    env.run(until=end_time)

    # View the results
    print(f'Average first layer length = {q_first_layer_sum / end_time}')
    print(f'Average Q1 length = {Q_second_layer_lens_sum[0] / end_time}')
    print(f'Average Q2 length = {Q_second_layer_lens_sum[1] / end_time}')
    print(f'Average Q3 length = {Q_second_layer_lens_sum[2] / end_time}')
    print(f'CPU efficiency = {cpu_working_time / end_time}')
    lane0_waiting_time = 0
    lane1_waiting_time = 0
    lane2_waiting_time = 0
    lane3_waiting_time = 0
    number_of_timed_outs = 0
    for j in Jobs:
        if j.starvation:
            number_of_timed_outs += 1
        if j.finish_time == 0 and not j.starvation:
            if j.lane == 0:
                j.waiting_time[0] += env.now - j.arrival_time
            elif j.lane == 1:
                j.waiting_time[1] += env.now - (j.arrival_time + j.waiting_time[0])
            elif j.lane == 2:
                j.waiting_time[2] += env.now - (j.arrival_time + j.waiting_time[0] + j.waiting_time[1] + T1)
            elif j.lane == 3:
                j.waiting_time[3] += env.now - (j.arrival_time + j.waiting_time[0] + j.waiting_time[1] + T1 + T2)

        lane0_waiting_time += j.waiting_time[0]
        lane1_waiting_time += j.waiting_time[1]
        lane2_waiting_time += j.waiting_time[2]
        lane3_waiting_time += j.waiting_time[3]

    print(f'Average Q0 waiting time = {lane0_waiting_time / len(Jobs)}')
    print(f'Average Q1 waiting time = {lane1_waiting_time / len(Jobs)}')
    print(f'Average Q2 waiting time = {lane2_waiting_time / len(Jobs)}')
    print(f'Average Q3 waiting time = {lane3_waiting_time / len(Jobs)}')
    print(f'The percentage of timed out processes = {number_of_timed_outs * 100 / len(Jobs)}')


if __name__ == "__main__":
    main()
