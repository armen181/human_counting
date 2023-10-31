from time import perf_counter, sleep


class FPSLimiter:
    def __init__(self, fps_cap: int = 120):
        self.fps_cap = fps_cap
        self.max_frame_duration = 1 / fps_cap
        self.previous_time = perf_counter()

    def update(self):
        frame_duration = perf_counter() - self.previous_time
        sleep_duration = max(0, self.max_frame_duration - frame_duration) * 0.8
        sleep(sleep_duration)
        self.previous_time = perf_counter()

