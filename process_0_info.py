def get_file_dev_id():
    file_path = "./audio/vad_example.wav"
    #file_path = "./audio/20240711090630019.wav"
    dev_id = 5
    model_dir = "bmodel/"
    #model_dir = "iic/"
    target = "BM1684X"
    return file_path, dev_id, model_dir, target
