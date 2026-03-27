def raw_transfer_time(data_size, bandwidth):
    return data_size / bandwidth


def turboquant_time(data_size, bandwidth, ratio, enc_speed, dec_speed):
    compressed_size = data_size * ratio
    
    t_transfer = compressed_size / bandwidth
    t_encode = data_size / enc_speed
    t_decode = data_size / dec_speed
    
    return t_transfer + t_encode + t_decode


def golden_cross(data_size, bandwidth, ratio, enc_speed, dec_speed):
    raw = raw_transfer_time(data_size, bandwidth)
    tq = turboquant_time(data_size, bandwidth, ratio, enc_speed, dec_speed)
    
    return tq < raw, raw, tq
