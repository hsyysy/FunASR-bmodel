from funasr import AutoModel
import time
import multiprocessing 

from process_0_info import get_file_dev_id

file_path, dev_id, model_dir = get_file_dev_id()

def process():
    # offline asr demo
    model = AutoModel(
            #model=model_dir+"speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",         ## 语音识别模型
            model=model_dir+"speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",      ## 语音识别模型
            vad_model=model_dir+"speech_fsmn_vad_zh-cn-16k-common-pytorch",                                  ## 语音端点检测模型
            punc_model=model_dir+"punc_ct-transformer_zh-cn-common-vocab272727-pytorch",                     ## 标点恢复模型
            spk_model=model_dir+"speech_campplus_sv_zh-cn_16k-common",                                       ## 说话人识别模型
            device="cpu",
            disable_update=True,
            disable_pbar=True,
            dev_id=dev_id,
    )
    # inference
    start_time = time.time()
    res = model.generate(input=file_path, batch_size_s=300,)
    end_time = time.time()
    #print(res[0]["text"])
    for si in res[0]["sentence_info"]:
        print("["+format(si["start"],"7d")+"]["+format(si["end"],"7d")+"], spk="+format(si["spk"],'2d')+", text: "+si["text"])
    print("generate time:", end_time-start_time)

"""
ps = []
for _ in range(1):
    p = multiprocessing.Process(target=process)
    p.start()
    ps.append(p)

for p in ps:
    p.join()
"""
process()
