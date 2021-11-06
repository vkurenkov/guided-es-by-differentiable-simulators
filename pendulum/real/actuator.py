import can
import numpy as np
import time

class State:
    def __init__(self):
        self.time = 0
        self.current = 0
        self.X = np.zeros((2, 1))


class Actuator:
    def __init__(self, id, bus):
        self.rev_n = 0
        self.id = id
        self.bus = bus
        self.state = State()
        self.start_time = time.time()
        self.cur_scale = 33/2048
        self.vel_scale = 2*np.pi/16384*32*3/2
        self.pos_scale = 2*np.pi/16384
        self.pos_offset = 8175#6198

    def __del__(self):
        self.send_current(0)

    def act_time(self):
        return time.time() - self.state.time - self.start_time

    def recv_can_answ(self):
        res = self.bus.recv()
        res_ar = [x for x in res.data]

        cur = res_ar[2] + (res_ar[3] << 8)
        self.state.current = (cur-(cur >> 15)*65536)*self.cur_scale

        vel = res_ar[4] + (res_ar[5] << 8)
        vel = (vel-(vel >> 15)*65536)*self.vel_scale
        self.state.X[1] = vel

        new_pos = (res_ar[6] + (res_ar[7] << 8) -
                   self.pos_offset)*self.pos_scale
        # print(res_ar[6] + (res_ar[7] << 8))
        rel_pos = self.state.X[0] - self.rev_n*2*np.pi
        if abs(new_pos - rel_pos) > np.pi and np.sign(vel) == np.sign(rel_pos - new_pos):
            #print(new_pos,self.state.X[0] - self.rev_n*2*np.pi)
            self.rev_n = self.rev_n - np.sign(new_pos - rel_pos)
        vel = (new_pos + self.rev_n*2*np.pi -
               self.state.X[0])/(time.time() - self.state.time)
        self.state.time = time.time() - self.start_time
        self.state.X[0] = new_pos + self.rev_n*2*np.pi

        return self.state

    def send_current(self, current):
        current = int(current/self.cur_scale)
        current = -((current >> 15)*2**16 - current)
        msg = can.Message(arbitration_id=self.id,
                          data=[0xA1, 0, 0, 0, current &
                                0x00FF, current >> 8, 0, 0],
                          is_extended_id=False)
        self.bus.send(msg)
        state = self.recv_can_answ()
        return state

    def get_state(self):
        msg = can.Message(arbitration_id=self.id,
                          data=[0x9c, 0, 0, 0, 0, 0, 0, 0],
                          is_extended_id=False)
        self.bus.send(msg)
        state = self.recv_can_answ()
        return state


class Encoder:
    def __init__(self):
        self.spi = spidev.SpiDev()
        self.spi.open(0, 1)
        self.spi.max_speed_hz = 3000000
        self.spi.mode = 0b00
        self.state = State()
        self.rev_n = 0
        self.cpr = 524287
        self.pos_scale = 2*np.pi/self.cpr
        self.start_time = time.time()
        self.pos_offset = -175768

    def get_state(self):
        arr = self.spi.readbytes(5)
        bit_pos = arr[0]
        bit_pos = bit_pos << 8
        bit_pos = bit_pos | arr[1]
        bit_pos = bit_pos << 4
        bit_pos = bit_pos | (arr[2] >> 4)
        bit_pos = -bit_pos
        # print(bit_pos)
        new_pos = (bit_pos - self.pos_offset)*self.pos_scale
        rel_pos = self.state.X[0] - self.rev_n * 2 * np.pi
        if abs(new_pos - rel_pos) > np.pi:
            self.rev_n = self.rev_n - np.sign(new_pos - rel_pos)

        vel = (new_pos + self.rev_n*2*np.pi -
               self.state.X[0]) / (time.time() - self.state.time - self.start_time)

        self.state.X[0] = new_pos + self.rev_n * 2 * np.pi
        self.state.X[1] = vel
        self.state.time = time.time() - self.start_time
        return self.state
