"""Constants fixed by the Pryon model generation pypryon supports."""

SAMPLE_RATE = 16000
FRAME_SAMPLES = 160   # 10 ms LFBE hop
CYCLE_FRAMES = 6      # LFBE frames per graph step (kw.cfg scorer.score_upsampling)
LANES = 25            # LSTM ring lanes, each a <=1.5 s sliding window
