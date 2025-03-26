/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#include "precomp.h"
#include "paraformer.h"
#include "encode_converter.h"
#include <cstddef>

using namespace std;
namespace funasr {

Paraformer::Paraformer()
:use_hotword(false),
 env_(ORT_LOGGING_LEVEL_ERROR, "paraformer"),session_options_{},
 hw_env_(ORT_LOGGING_LEVEL_ERROR, "paraformer_hw"),hw_session_options{} {
    status = bm_dev_request(&bm_handle, DEV_ID);
    assert(BM_SUCCESS == status);

    unsigned p_chipid;
    bm_get_chipid(bm_handle, &p_chipid);
    if (p_chipid == 0x1686a200)
        is_1688 = true;
}

// bmrt
void Paraformer::InitBmrt(const char* en_model, const char* de_model){
    try {
        p_bmrt_offline_encoder = bmrt_create(bm_handle);
        assert(NULL != p_bmrt_offline_encoder);
        ret = bmrt_load_bmodel(p_bmrt_offline_encoder, en_model);
        assert(true == ret);
        LOG(INFO) << "Successfully load encoder model from " << en_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load encoder bmodel: " << e.what();
        exit(-1);
    }

    try {
        p_bmrt_offline_decoder = bmrt_create(bm_handle);
        assert(NULL != p_bmrt_offline_decoder);
        ret = bmrt_load_bmodel(p_bmrt_offline_decoder, de_model);
        assert(true == ret);
        LOG(INFO) << "Successfully load decoder model from " << de_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load decoder bmodel: " << e.what();
        exit(-1);
    }
}

// offline
void Paraformer::InitAsr(const std::string &am_model, const std::string &am_cmvn, const std::string &am_config, const std::string &token_file, int thread_num){
    LoadConfigFromYaml(am_config.c_str());
    // knf options
    fbank_opts_.frame_opts.dither = 0;
    fbank_opts_.mel_opts.num_bins = n_mels;
    fbank_opts_.frame_opts.samp_freq = asr_sample_rate;
    fbank_opts_.frame_opts.window_type = window_type;
    fbank_opts_.frame_opts.frame_shift_ms = frame_shift;
    fbank_opts_.frame_opts.frame_length_ms = frame_length;
    fbank_opts_.energy_floor = 0;
    fbank_opts_.mel_opts.debug_mel = false;
    // fbank_ = std::make_unique<knf::OnlineFbank>(fbank_opts);

    constexpr size_t ENCODER_MODEL_NAME_LENGTH = sizeof(ENCODER_MODEL_NAME) - 1;
    std::string model_dir = am_model.substr(0, am_model.size() - ENCODER_MODEL_NAME_LENGTH);
    std::string encoder_model = PathAppend(model_dir, ENCODER_MODEL_NAME);
    std::string decoder_model = PathAppend(model_dir, DECODER_MODEL_NAME);
    InitBmrt(encoder_model.c_str(), decoder_model.c_str());
    /*
    // session_options_.SetInterOpNumThreads(1);
    session_options_.SetIntraOpNumThreads(thread_num);
    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // DisableCpuMemArena can improve performance
    session_options_.DisableCpuMemArena();

    try {
        m_session_ = std::make_unique<Ort::Session>(env_, ORTSTRING(am_model).c_str(), session_options_);
        LOG(INFO) << "Successfully load model from " << am_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load am onnx model: " << e.what();
        exit(-1);
    }

    GetInputNames(m_session_.get(), m_strInputNames, m_szInputNames);
    GetOutputNames(m_session_.get(), m_strOutputNames, m_szOutputNames);
    */
    vocab = new Vocab(token_file.c_str());
	phone_set_ = new PhoneSet(token_file.c_str());
    LoadCmvn(am_cmvn.c_str());
}

// online
void Paraformer::InitAsr(const std::string &en_model, const std::string &de_model, const std::string &am_cmvn, const std::string &am_config, const std::string &token_file, int thread_num){
    
    LoadOnlineConfigFromYaml(am_config.c_str());
    // knf options
    fbank_opts_.frame_opts.dither = 0;
    fbank_opts_.mel_opts.num_bins = n_mels;
    fbank_opts_.frame_opts.samp_freq = asr_sample_rate;
    fbank_opts_.frame_opts.window_type = window_type;
    fbank_opts_.frame_opts.frame_shift_ms = frame_shift;
    fbank_opts_.frame_opts.frame_length_ms = frame_length;
    fbank_opts_.energy_floor = 0;
    fbank_opts_.mel_opts.debug_mel = false;

    // session_options_.SetInterOpNumThreads(1);
    session_options_.SetIntraOpNumThreads(thread_num);
    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // DisableCpuMemArena can improve performance
    session_options_.DisableCpuMemArena();

    constexpr size_t ENCODER_ONLINE_MODEL_NAME_LENGTH = sizeof(ENCODER_ONLINE_MODEL_NAME) - 1;
    std::string model_dir = en_model.substr(0, en_model.size() - ENCODER_ONLINE_MODEL_NAME_LENGTH);
    std::string encoder_model = PathAppend(model_dir, ENCODER_ONLINE_MODEL_NAME);
    std::string decoder_model = PathAppend(model_dir, DECODER_ONLINE_MODEL_NAME);

    try {
        //encoder_session_ = std::make_unique<Ort::Session>(env_, ORTSTRING(en_model).c_str(), session_options_);
        p_bmrt_online_encoder = bmrt_create(bm_handle);
        assert(NULL != p_bmrt_online_encoder);
        ret = bmrt_load_bmodel(p_bmrt_online_encoder, encoder_model.c_str());
        assert(true == ret);
        LOG(INFO) << "Successfully load bmodel from " << encoder_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load am encoder bmodel: " << e.what();
        exit(-1);
    }

    try {
        //decoder_session_ = std::make_unique<Ort::Session>(env_, ORTSTRING(de_model).c_str(), session_options_);
        p_bmrt_online_decoder = bmrt_create(bm_handle);
        assert(NULL != p_bmrt_online_decoder);
        ret = bmrt_load_bmodel(p_bmrt_online_decoder, decoder_model.c_str());
        assert(true == ret);
        LOG(INFO) << "Successfully load bmodel from " << de_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load am decoder model: " << e.what();
        exit(-1);
    }

    // encoder
    string strName;
    /*
    GetInputName(encoder_session_.get(), strName);
    en_strInputNames.push_back(strName.c_str());
    GetInputName(encoder_session_.get(), strName,1);
    en_strInputNames.push_back(strName);
    
    GetOutputName(encoder_session_.get(), strName);
    en_strOutputNames.push_back(strName);
    GetOutputName(encoder_session_.get(), strName,1);
    en_strOutputNames.push_back(strName);
    GetOutputName(encoder_session_.get(), strName,2);
    en_strOutputNames.push_back(strName);

    for (auto& item : en_strInputNames)
        en_szInputNames_.push_back(item.c_str());
    for (auto& item : en_strOutputNames)
        en_szOutputNames_.push_back(item.c_str());

    // decoder
    int de_input_len = 4 + fsmn_layers;
    int de_out_len = 2 + fsmn_layers;
    for(int i=0;i<de_input_len; i++){
        GetInputName(decoder_session_.get(), strName, i);
        de_strInputNames.push_back(strName.c_str());
    }

    for(int i=0;i<de_out_len; i++){
        GetOutputName(decoder_session_.get(), strName,i);
        de_strOutputNames.push_back(strName);
    }

    for (auto& item : de_strInputNames)
        de_szInputNames_.push_back(item.c_str());
    for (auto& item : de_strOutputNames)
        de_szOutputNames_.push_back(item.c_str());
    */

    vocab = new Vocab(token_file.c_str());
    phone_set_ = new PhoneSet(token_file.c_str());
    LoadCmvn(am_cmvn.c_str());
}

// 2pass
void Paraformer::InitAsr(const std::string &am_model, const std::string &en_model, const std::string &de_model, 
    const std::string &am_cmvn, const std::string &am_config, const std::string &token_file, const std::string &online_token_file, int thread_num){
    // online
    InitAsr(en_model, de_model, am_cmvn, am_config, online_token_file, thread_num);

    // offline
    constexpr size_t ENCODER_MODEL_NAME_LENGTH = sizeof(ENCODER_MODEL_NAME) - 1;
    std::string model_dir = am_model.substr(0, am_model.size() - ENCODER_MODEL_NAME_LENGTH);
    std::string encoder_model = PathAppend(model_dir, ENCODER_MODEL_NAME);
    std::string decoder_model = PathAppend(model_dir, DECODER_MODEL_NAME);
    InitBmrt(encoder_model.c_str(), decoder_model.c_str());
    /*
    try {
        m_session_ = std::make_unique<Ort::Session>(env_, ORTSTRING(am_model).c_str(), session_options_);
        LOG(INFO) << "Successfully load model from " << am_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load am onnx model: " << e.what();
        exit(-1);
    }

    GetInputNames(m_session_.get(), m_strInputNames, m_szInputNames);
    GetOutputNames(m_session_.get(), m_strOutputNames, m_szOutputNames);
    */
}

void Paraformer::InitLm(const std::string &lm_file, 
                        const std::string &lm_cfg_file, 
                        const std::string &lex_file) {
    try {
        lm_ = std::shared_ptr<fst::Fst<fst::StdArc>>(
            fst::Fst<fst::StdArc>::Read(lm_file));
        if (lm_){
            lm_vocab = new Vocab(lm_cfg_file.c_str(), lex_file.c_str());
            LOG(INFO) << "Successfully load lm file " << lm_file;
        }else{
            LOG(ERROR) << "Failed to load lm file " << lm_file;
        }
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load lm file: " << e.what();
        exit(0);
    }
}

void Paraformer::LoadConfigFromYaml(const char* filename){

    YAML::Node config;
    try{
        config = YAML::LoadFile(filename);
    }catch(exception const &e){
        LOG(ERROR) << "Error loading file, yaml file error or not exist.";
        exit(-1);
    }

    try{
        YAML::Node frontend_conf = config["frontend_conf"];
        this->asr_sample_rate = frontend_conf["fs"].as<int>();

        YAML::Node lang_conf = config["lang"];
        if (lang_conf.IsDefined()){
            language = lang_conf.as<string>();
        }
    }catch(exception const &e){
        LOG(ERROR) << "Error when load argument from vad config YAML.";
        exit(-1);
    }
}

void Paraformer::LoadOnlineConfigFromYaml(const char* filename){

    YAML::Node config;
    try{
        config = YAML::LoadFile(filename);
    }catch(exception const &e){
        LOG(ERROR) << "Error loading file, yaml file error or not exist.";
        exit(-1);
    }

    try{
        YAML::Node frontend_conf = config["frontend_conf"];
        YAML::Node encoder_conf = config["encoder_conf"];
        YAML::Node decoder_conf = config["decoder_conf"];
        YAML::Node predictor_conf = config["predictor_conf"];

        this->window_type = frontend_conf["window"].as<string>();
        this->n_mels = frontend_conf["n_mels"].as<int>();
        this->frame_length = frontend_conf["frame_length"].as<int>();
        this->frame_shift = frontend_conf["frame_shift"].as<int>();
        this->lfr_m = frontend_conf["lfr_m"].as<int>();
        this->lfr_n = frontend_conf["lfr_n"].as<int>();

        this->encoder_size = encoder_conf["output_size"].as<int>();
        this->fsmn_dims = encoder_conf["output_size"].as<int>();

        this->fsmn_layers = decoder_conf["num_blocks"].as<int>();
        this->fsmn_lorder = decoder_conf["kernel_size"].as<int>()-1;

        this->cif_threshold = predictor_conf["threshold"].as<double>();
        this->tail_alphas = predictor_conf["tail_threshold"].as<double>();

        this->asr_sample_rate = frontend_conf["fs"].as<int>();


    }catch(exception const &e){
        LOG(ERROR) << "Error when load argument from vad config YAML.";
        exit(-1);
    }
}

void Paraformer::InitHwCompiler(const std::string &hw_model, int thread_num) {
    /*
    hw_session_options.SetIntraOpNumThreads(thread_num);
    hw_session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // DisableCpuMemArena can improve performance
    hw_session_options.DisableCpuMemArena();
    */

    try {
        //hw_m_session = std::make_unique<Ort::Session>(hw_env_, ORTSTRING(hw_model).c_str(), hw_session_options);
        status = bm_dev_request(&bm_handle, DEV_ID);
        assert(BM_SUCCESS == status);
        p_bmrt_hw = bmrt_create(bm_handle);
        assert(NULL != p_bmrt_hw);
        ret = bmrt_load_bmodel(p_bmrt_hw, hw_model.c_str());
        assert(true == ret);
        LOG(INFO) << "Successfully load model from " << hw_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load hw compiler bmodel: " << e.what();
        exit(-1);
    }

    /*
    string strName;
    GetInputName(hw_m_session.get(), strName);
    hw_m_strInputNames.push_back(strName.c_str());
    //GetInputName(hw_m_session.get(), strName,1);
    //hw_m_strInputNames.push_back(strName);
    
    GetOutputName(hw_m_session.get(), strName);
    hw_m_strOutputNames.push_back(strName);

    for (auto& item : hw_m_strInputNames)
        hw_m_szInputNames.push_back(item.c_str());
    for (auto& item : hw_m_strOutputNames)
        hw_m_szOutputNames.push_back(item.c_str());
    */
    // if init hotword compiler is called, this is a hotword paraformer model
    use_hotword = true;
}

void Paraformer::InitSegDict(const std::string &seg_dict_model) {
    seg_dict = new SegDict(seg_dict_model.c_str());
}

Paraformer::~Paraformer()
{
    if(vocab){
        delete vocab;
    }
    if(lm_vocab){
        delete lm_vocab;
    }
    if(seg_dict){
        delete seg_dict;
    }
    if(phone_set_){
        delete phone_set_;
    }
    if(p_bmrt_offline_encoder!=NULL){
        bmrt_destroy(p_bmrt_offline_encoder);
    }
    if(p_bmrt_offline_decoder!=NULL){
        bmrt_destroy(p_bmrt_offline_decoder);
    }
    if(p_bmrt_online_encoder!=NULL){
        bmrt_destroy(p_bmrt_online_encoder);
    }
    if(p_bmrt_online_decoder!=NULL){
        bmrt_destroy(p_bmrt_online_decoder);
    }
    if(p_bmrt_hw!=NULL){
        bmrt_destroy(p_bmrt_hw);
    }
    if(bm_handle){
        bm_dev_free(bm_handle);
    }
}

void Paraformer::StartUtterance()
{
}

void Paraformer::EndUtterance()
{
}

void Paraformer::Reset()
{
}

void Paraformer::FbankKaldi(float sample_rate, const float* waves, int len, std::vector<std::vector<float>> &asr_feats) {
    knf::OnlineFbank fbank_(fbank_opts_);
    std::vector<float> buf(len);
    for (int32_t i = 0; i != len; ++i) {
        buf[i] = waves[i] * 32768;
    }
    fbank_.AcceptWaveform(sample_rate, buf.data(), buf.size());

    int32_t frames = fbank_.NumFramesReady();
    for (int32_t i = 0; i != frames; ++i) {
        const float *frame = fbank_.GetFrame(i);
        std::vector<float> frame_vector(frame, frame + fbank_opts_.mel_opts.num_bins);
        asr_feats.emplace_back(frame_vector);
    }
}

void Paraformer::LoadCmvn(const char *filename)
{
    ifstream cmvn_stream(filename);
    if (!cmvn_stream.is_open()) {
        LOG(ERROR) << "Failed to open file: " << filename;
        exit(-1);
    }
    string line;

    while (getline(cmvn_stream, line)) {
        istringstream iss(line);
        vector<string> line_item{istream_iterator<string>{iss}, istream_iterator<string>{}};
        if (line_item[0] == "<AddShift>") {
            getline(cmvn_stream, line);
            istringstream means_lines_stream(line);
            vector<string> means_lines{istream_iterator<string>{means_lines_stream}, istream_iterator<string>{}};
            if (means_lines[0] == "<LearnRateCoef>") {
                for (int j = 3; j < means_lines.size() - 1; j++) {
                    means_list_.push_back(stof(means_lines[j]));
                }
                continue;
            }
        }
        else if (line_item[0] == "<Rescale>") {
            getline(cmvn_stream, line);
            istringstream vars_lines_stream(line);
            vector<string> vars_lines{istream_iterator<string>{vars_lines_stream}, istream_iterator<string>{}};
            if (vars_lines[0] == "<LearnRateCoef>") {
                for (int j = 3; j < vars_lines.size() - 1; j++) {
                    vars_list_.push_back(stof(vars_lines[j])*scale);
                }
                continue;
            }
        }
    }
}

string Paraformer::GreedySearch(float * in, int n_len,  int64_t token_nums, bool is_stamp, std::vector<float> us_alphas, std::vector<float> us_cif_peak)
{
    vector<int> hyps;
    int Tmax = n_len;
    for (int i = 0; i < Tmax; i++) {
        int max_idx;
        float max_val;
        FindMax(in + i * token_nums, token_nums, max_val, max_idx);
        hyps.push_back(max_idx);
    }
    if(!is_stamp){
        return vocab->Vector2StringV2(hyps, language);
    }else{
        std::vector<string> char_list;
        std::vector<std::vector<float>> timestamp_list;
        std::string res_str;
        vocab->Vector2String(hyps, char_list);
        std::vector<string> raw_char(char_list);
        TimestampOnnx(us_alphas, us_cif_peak, char_list, res_str, timestamp_list);

        return PostProcess(raw_char, timestamp_list);
    }
}

string Paraformer::BeamSearch(WfstDecoder* &wfst_decoder, float *in, int len, int64_t token_nums)
{
  return wfst_decoder->Search(in, len, token_nums);
}

string Paraformer::FinalizeDecode(WfstDecoder* &wfst_decoder,
                                  bool is_stamp, std::vector<float> us_alphas, std::vector<float> us_cif_peak)
{
  return wfst_decoder->FinalizeDecode(is_stamp, us_alphas, us_cif_peak);
}

void Paraformer::LfrCmvn(std::vector<std::vector<float>> &asr_feats) {

    std::vector<std::vector<float>> out_feats;
    int T = asr_feats.size();
    int T_lrf = ceil(1.0 * T / lfr_n);

    // Pad frames at start(copy first frame)
    for (int i = 0; i < (lfr_m - 1) / 2; i++) {
        asr_feats.insert(asr_feats.begin(), asr_feats[0]);
    }
    // Merge lfr_m frames as one,lfr_n frames per window
    T = T + (lfr_m - 1) / 2;
    std::vector<float> p;
    for (int i = 0; i < T_lrf; i++) {
        if (lfr_m <= T - i * lfr_n) {
            for (int j = 0; j < lfr_m; j++) {
                p.insert(p.end(), asr_feats[i * lfr_n + j].begin(), asr_feats[i * lfr_n + j].end());
            }
            out_feats.emplace_back(p);
            p.clear();
        } else {
            // Fill to lfr_m frames at last window if less than lfr_m frames  (copy last frame)
            int num_padding = lfr_m - (T - i * lfr_n);
            for (int j = 0; j < (asr_feats.size() - i * lfr_n); j++) {
                p.insert(p.end(), asr_feats[i * lfr_n + j].begin(), asr_feats[i * lfr_n + j].end());
            }
            for (int j = 0; j < num_padding; j++) {
                p.insert(p.end(), asr_feats[asr_feats.size() - 1].begin(), asr_feats[asr_feats.size() - 1].end());
            }
            out_feats.emplace_back(p);
            p.clear();
        }
    }
    // Apply cmvn
    for (auto &out_feat: out_feats) {
        for (int j = 0; j < means_list_.size(); j++) {
            out_feat[j] = (out_feat[j] + means_list_[j]) * vars_list_[j];
        }
    }
    asr_feats = out_feats;
}

std::vector<std::string> Paraformer::Forward(float** din, int* len, bool input_finished, const std::vector<std::vector<float>> &hw_emb, void* decoder_handle, int batch_in)
{
    std::vector<std::string> results;
    string result="";
    WfstDecoder* wfst_decoder = (WfstDecoder*)decoder_handle;
    int32_t in_feat_dim = fbank_opts_.mel_opts.num_bins;

    if(batch_in != 1){
        results.push_back(result);
        return results;
    }

    std::vector<std::vector<float>> asr_feats;
    FbankKaldi(asr_sample_rate, din[0], len[0], asr_feats);
    if(asr_feats.size() == 0){
        results.push_back(result);
        return results;
    }
    LfrCmvn(asr_feats);
    int32_t feat_dim = lfr_m*in_feat_dim;
    int32_t num_frames = asr_feats.size();

    std::vector<float> wav_feats;
    for (const auto &frame_feat: asr_feats) {
        wav_feats.insert(wav_feats.end(), frame_feat.begin(), frame_feat.end());
    }

    /*
#ifdef _WIN_X86
        Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
#else
        Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif

    const int64_t input_shape_[3] = {1, num_frames, feat_dim};
    Ort::Value onnx_feats = Ort::Value::CreateTensor<float>(m_memoryInfo,
        wav_feats.data(),
        wav_feats.size(),
        input_shape_,
        3);

    const int64_t paraformer_length_shape[1] = {1};
    std::vector<int32_t> paraformer_length;
    paraformer_length.emplace_back(num_frames);
    Ort::Value onnx_feats_len = Ort::Value::CreateTensor<int32_t>(
          m_memoryInfo, paraformer_length.data(), paraformer_length.size(), paraformer_length_shape, 1);

    std::vector<Ort::Value> input_onnx;
    input_onnx.emplace_back(std::move(onnx_feats));
    input_onnx.emplace_back(std::move(onnx_feats_len));

    std::vector<float> embedding;
    try{
        if (use_hotword) {
            if(hw_emb.size()<=0){
                LOG(ERROR) << "hw_emb is null";
                results.push_back(result);
                return results;
            }
            //PrintMat(hw_emb, "input_clas_emb");
            const int64_t hotword_shape[3] = {1, static_cast<int64_t>(hw_emb.size()), static_cast<int64_t>(hw_emb[0].size())};
            embedding.reserve(hw_emb.size() * hw_emb[0].size());
            for (auto item : hw_emb) {
                embedding.insert(embedding.end(), item.begin(), item.end());
            }
            //LOG(INFO) << "hotword shape " << hotword_shape[0] << " " << hotword_shape[1] << " " << hotword_shape[2] << " size " << embedding.size();
            Ort::Value onnx_hw_emb = Ort::Value::CreateTensor<float>(
                m_memoryInfo, embedding.data(), embedding.size(), hotword_shape, 3);

            input_onnx.emplace_back(std::move(onnx_hw_emb));
        }
    }catch (std::exception const &e)
    {
        LOG(ERROR)<<e.what();
        results.push_back(result);
        return results;
    }
    */

    try {
        net_names = NULL;
        bmrt_get_network_names(p_bmrt_offline_encoder, &net_names);
        net_info = bmrt_get_network_info(p_bmrt_offline_encoder, net_names[0]);
        assert(NULL != net_info);

        // input tensor of encoder
        bm_tensor_t input_tensors_encoder[net_info->input_num];
        input_tensors_encoder[0].shape = {3, {1, num_frames, feat_dim}};
        input_tensors_encoder[1].shape = {1, {1}};
        for (int i=0;i<net_info->input_num;i++){
            input_tensors_encoder[i].dtype = net_info->input_dtypes[i];
            if (is_1688) {
                size_t size = bmrt_tensor_bytesize(&input_tensors_encoder[i]);
                bm_malloc_device_byte(bm_handle, &input_tensors_encoder[i].device_mem, size);
            } else
                input_tensors_encoder[i].device_mem = net_info->stages[0].input_mems[i];
            input_tensors_encoder[i].st_mode = BM_STORE_1N;
        }

        status = bm_memcpy_s2d_partial(bm_handle, input_tensors_encoder[0].device_mem, wav_feats.data(), wav_feats.size()*sizeof(float));
        assert(BM_SUCCESS == status);

        status = bm_memcpy_s2d_partial(bm_handle, input_tensors_encoder[1].device_mem, &num_frames, sizeof(int32_t));
        assert(BM_SUCCESS == status);

        // output tensor of encoder
        bm_tensor_t output_tensors_encoder[net_info->output_num];
        output_tensors_encoder[0].shape = {3, {1,num_frames,512}};
        output_tensors_encoder[1].shape = {3, {1,num_frames+1,512}};
        output_tensors_encoder[2].shape = {2, {1,num_frames+1}};
        output_tensors_encoder[3].shape = {1, {1}};
        for (int i=0;i<net_info->output_num;i++){
            output_tensors_encoder[i].dtype = net_info->output_dtypes[i];
            if (is_1688) {
                size_t size = bmrt_tensor_bytesize(&output_tensors_encoder[i]);
                bm_malloc_device_byte(bm_handle, &output_tensors_encoder[i].device_mem, size);
            }else
                output_tensors_encoder[i].device_mem = net_info->stages[0].output_mems[i];
        }

        // forward
        ret = bmrt_launch_tensor_ex(p_bmrt_offline_encoder, net_names[0], input_tensors_encoder, 2, output_tensors_encoder, 4, true, false);
        assert(true == ret);
        bm_thread_sync(bm_handle);

        // hidden
        output_data.resize(bmrt_shape_count(&output_tensors_encoder[1].shape));
        status = bm_memcpy_d2s_partial(bm_handle, output_data.data(), output_tensors_encoder[1].device_mem, bmrt_tensor_bytesize(&output_tensors_encoder[1]));
        assert(BM_SUCCESS == status);
        int batch_size = output_tensors_encoder[1].shape.dims[0];
        int time_steps = output_tensors_encoder[1].shape.dims[1];
        int hidden_size = output_tensors_encoder[1].shape.dims[2];
        std::vector<std::vector<std::vector<float>>> hidden(batch_size, std::vector<std::vector<float>>(time_steps, std::vector<float>(hidden_size)));
        auto output_it = output_data.begin();
        for (auto& batch : hidden) {
            for (auto& time_step : batch) {
                std::copy(output_it, output_it + hidden_size, time_step.begin());
                output_it += hidden_size;
            }
        }

        // alphas
        output_data.resize(bmrt_shape_count(&output_tensors_encoder[2].shape));
        status = bm_memcpy_d2s_partial(bm_handle, output_data.data(), output_tensors_encoder[2].device_mem, bmrt_tensor_bytesize(&output_tensors_encoder[2]));
        assert(BM_SUCCESS == status);
        std::vector<std::vector<float>> alphas;
        batch_size = output_tensors_encoder[2].shape.dims[0];
        time_steps = output_tensors_encoder[2].shape.dims[1];
        alphas.resize(batch_size);
        for (int b = 0; b < batch_size; ++b) {
            alphas[b].resize(time_steps);
            for (int t = 0; t < time_steps; ++t) {
                alphas[b][t] = output_data[b * time_steps + t];
            }
        }

        // pre_token_length
        output_data.resize(bmrt_shape_count(&output_tensors_encoder[3].shape));
        status = bm_memcpy_d2s_partial(bm_handle, output_data.data(), output_tensors_encoder[3].device_mem, bmrt_tensor_bytesize(&output_tensors_encoder[3]));
        assert(BM_SUCCESS == status);
        auto token_num_int = static_cast<int>(*std::max_element(output_data.begin(), output_data.end()));

        std::vector<long> pre_token_length_rounded;
        for (int i = 0; i < output_data.size(); i++){
            pre_token_length_rounded.push_back(static_cast<long>(std::round(output_data[i])));
        }

        // before next bmodel
        auto cif_result = cif(hidden, alphas, 1.0f);

        auto pre_acoustic_embeds = cif_result.first;
        auto cif_peak = cif_result.second;
        int feature_size = pre_acoustic_embeds[0][0].size();
        std::vector<float> pre_acoustic_embeds2;
        pre_acoustic_embeds2.reserve(batch_size*token_num_int*feature_size);
        for (const auto& matrix : pre_acoustic_embeds) {
            for (int i = 0; i < token_num_int; ++i) {
                pre_acoustic_embeds2.insert(pre_acoustic_embeds2.end(), matrix[i].begin(), matrix[i].end());
            }
        }

        int hw_emb_count = static_cast<int>(hw_emb.size());
        std::vector<float> hw_embed;
        hw_embed.reserve(hw_emb_count);
        for (const auto& row : hw_emb) {
            hw_embed.insert(hw_embed.end(), row.begin(), row.end());
        }

        // decoder
        free(net_names);
        net_names = NULL;
        bmrt_get_network_names(p_bmrt_offline_decoder, &net_names);
        net_info = bmrt_get_network_info(p_bmrt_offline_decoder, net_names[0]);
        assert(NULL != net_info);

        bm_tensor_t input_tensors_decoder[net_info->input_num];
        input_tensors_decoder[0].shape = {3, {batch_size, num_frames, 512}};
        input_tensors_decoder[1].shape = {1, {batch_size}};
        input_tensors_decoder[2].shape = {3, {batch_size, token_num_int, feature_size}};
        input_tensors_decoder[3].shape = {1, {batch_size}};
        input_tensors_decoder[4].shape = {3, {batch_size, hw_emb_count, 512}};
        for(int i=0;i<net_info->input_num;i++){
            input_tensors_decoder[i].dtype = net_info->input_dtypes[i];
            if (i>1) {
                if (is_1688) {
                    size_t size = bmrt_tensor_bytesize(&input_tensors_decoder[i]);
                    bm_malloc_device_byte(bm_handle, &input_tensors_decoder[i].device_mem, size);
                } else {
                    input_tensors_decoder[i].device_mem = net_info->stages[0].input_mems[i];
                }
            }
            input_tensors_decoder[i].st_mode = BM_STORE_1N;
        }
        input_tensors_decoder[0].device_mem = output_tensors_encoder[0].device_mem;
        input_tensors_decoder[1].device_mem = input_tensors_encoder[1].device_mem;

        status = bm_memcpy_s2d_partial(bm_handle, input_tensors_decoder[2].device_mem, pre_acoustic_embeds2.data(), pre_acoustic_embeds2.size()*sizeof(float));
        assert(BM_SUCCESS == status);

        status = bm_memcpy_s2d_partial(bm_handle, input_tensors_decoder[3].device_mem, pre_token_length_rounded.data(), batch_size*sizeof(int32_t));
        assert(BM_SUCCESS == status);

        status = bm_memcpy_s2d_partial(bm_handle, input_tensors_decoder[4].device_mem, hw_embed.data(), batch_size*hw_emb_count*512*sizeof(float));
        assert(BM_SUCCESS == status);

        bm_tensor_t output_tensors_decoder[1];
        if (is_1688){
            status = bm_malloc_device_byte(bm_handle, &output_tensors_decoder[0].device_mem, net_info->max_output_bytes[0]);
            assert(BM_SUCCESS == status);
        } else
            output_tensors_decoder[0].device_mem = net_info->stages[0].output_mems[0];

        ret = bmrt_launch_tensor_ex(p_bmrt_offline_decoder, net_names[0], input_tensors_decoder, 5, output_tensors_decoder, 1, true, false);
        assert(true == ret);
        bm_thread_sync(bm_handle);

        auto decoder_out_shape = output_tensors_decoder[0].shape;
        output_data.resize(bmrt_shape_count(&decoder_out_shape));
        status = bm_memcpy_d2s_partial(bm_handle, output_data.data(), output_tensors_decoder[0].device_mem, bmrt_tensor_bytesize(&output_tensors_decoder[0]));
        assert(BM_SUCCESS == status);
        auto decoder_out_lens = decoder_out_shape.dims[1];
        auto decoder_out_vocab = decoder_out_shape.dims[2];

        // free device memory
        if (is_1688){
            // free encoder
            free(net_names);
            net_names = NULL;
            bmrt_get_network_names(p_bmrt_offline_encoder, &net_names);
            net_info = bmrt_get_network_info(p_bmrt_offline_encoder, net_names[0]);
            assert(NULL != net_info);
            for (int i = 0; i < net_info->input_num; ++i) {
                bm_free_device(bm_handle, input_tensors_encoder[i].device_mem);
            }
            for (int i = 0; i < net_info->output_num; ++i) {
                bm_free_device(bm_handle, output_tensors_encoder[i].device_mem);
            }
            // free decoder
            free(net_names);
            net_names = NULL;
            bmrt_get_network_names(p_bmrt_offline_decoder, &net_names);
            net_info = bmrt_get_network_info(p_bmrt_offline_decoder, net_names[0]);
            assert(NULL != net_info);
            for (int i = 2; i < net_info->input_num; ++i) {
                bm_free_device(bm_handle, input_tensors_decoder[i].device_mem);
            }
            for (int i = 0; i < net_info->output_num; ++i) {
                bm_free_device(bm_handle, output_tensors_decoder[i].device_mem);
            }
        }
        free(net_names);

        result = BeamSearch(wfst_decoder, output_data.data(), decoder_out_lens, decoder_out_vocab);
        if (input_finished) {
            result = FinalizeDecode(wfst_decoder);
        }

        /*
        auto outputTensor = m_session_->Run(Ort::RunOptions{nullptr}, m_szInputNames.data(), input_onnx.data(), input_onnx.size(), m_szOutputNames.data(), m_szOutputNames.size());
        std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
        //LOG(INFO) << "paraformer out shape " << outputShape[0] << " " << outputShape[1] << " " << outputShape[2];

        int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
        float* floatData = outputTensor[0].GetTensorMutableData<float>();
        auto encoder_out_lens = outputTensor[1].GetTensorMutableData<int64_t>();
        // timestamp
        if(outputTensor.size() == 4){
            std::vector<int64_t> us_alphas_shape = outputTensor[2].GetTensorTypeAndShapeInfo().GetShape();
            float* us_alphas_data = outputTensor[2].GetTensorMutableData<float>();
            std::vector<float> us_alphas(us_alphas_shape[1]);
            for (int i = 0; i < us_alphas_shape[1]; i++) {
                us_alphas[i] = us_alphas_data[i];
            }

            std::vector<int64_t> us_peaks_shape = outputTensor[3].GetTensorTypeAndShapeInfo().GetShape();
            float* us_peaks_data = outputTensor[3].GetTensorMutableData<float>();
            std::vector<float> us_peaks(us_peaks_shape[1]);
            for (int i = 0; i < us_peaks_shape[1]; i++) {
                us_peaks[i] = us_peaks_data[i];
            }
			if (lm_ == nullptr) {
                result = GreedySearch(floatData, *encoder_out_lens, outputShape[2], true, us_alphas, us_peaks);
			} else {
			    result = BeamSearch(wfst_decoder, floatData, *encoder_out_lens, outputShape[2]);
                if (input_finished) {
                    result = FinalizeDecode(wfst_decoder, true, us_alphas, us_peaks);
                }
			}
        }else{
			if (lm_ == nullptr) {
                result = GreedySearch(floatData, *encoder_out_lens, outputShape[2]);
			} else {
			    result = BeamSearch(wfst_decoder, floatData, *encoder_out_lens, outputShape[2]);
                if (input_finished) {
                    result = FinalizeDecode(wfst_decoder);
                }
			}
        }
        */
    }
    catch (std::exception const &e)
    {
        LOG(ERROR)<<e.what();
    }

    results.push_back(result);
    return results;
}


std::vector<std::vector<float>> Paraformer::CompileHotwordEmbedding(std::string &hotwords) {
    int embedding_dim = encoder_size;
    std::vector<std::vector<float>> hw_emb;
    if (!use_hotword) {
        std::vector<float> vec(embedding_dim, 0);
        hw_emb.push_back(vec);
        return hw_emb;
    }
    int max_hotword_len = 10;
    std::vector<int32_t> hotword_matrix;
    std::vector<int32_t> lengths;
    int hotword_size = 1;
    int real_hw_size = 0;
    if (!hotwords.empty()) {
      std::vector<std::string> hotword_array = split(hotwords, ' ');
      hotword_size = hotword_array.size() + 1;
      hotword_matrix.reserve(hotword_size * max_hotword_len);
      for (auto hotword : hotword_array) {
        std::vector<std::string> chars;
        if (EncodeConverter::IsAllChineseCharactor((const U8CHAR_T*)hotword.c_str(), hotword.size())) {
          KeepChineseCharacterAndSplit(hotword, chars);
        } else {
          // for english
          std::vector<std::string> words = split(hotword, ' ');
          for (auto word : words) {
            std::vector<string> tokens = seg_dict->GetTokensByWord(word);
            chars.insert(chars.end(), tokens.begin(), tokens.end());
          }
        }
        if(chars.size()==0){
            continue;
        }
        std::vector<int32_t> hw_vector(max_hotword_len, 0);
        int vector_len = std::min(max_hotword_len, (int)chars.size());
        int chs_oov = false;
        for (int i=0; i<vector_len; i++) {
          hw_vector[i] = phone_set_->String2Id(chars[i]);
          if(hw_vector[i] == -1){
            chs_oov = true;
            break;
          }
        }
        if(chs_oov){
          LOG(INFO) << "OOV: " << hotword;
          continue;
        }
        LOG(INFO) << hotword;
        lengths.push_back(vector_len);
        real_hw_size += 1;
        hotword_matrix.insert(hotword_matrix.end(), hw_vector.begin(), hw_vector.end());
      }
      hotword_size = real_hw_size + 1;
    }
    std::vector<int32_t> blank_vec(max_hotword_len, 0);
    blank_vec[0] = 1;
    hotword_matrix.insert(hotword_matrix.end(), blank_vec.begin(), blank_vec.end());
    lengths.push_back(1);

    /*
#ifdef _WIN_X86
        Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
#else
        Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif

    const int64_t input_shape_[2] = {hotword_size, max_hotword_len};
    Ort::Value onnx_hotword = Ort::Value::CreateTensor<int32_t>(m_memoryInfo,
        (int32_t*)hotword_matrix.data(),
        hotword_size * max_hotword_len,
        input_shape_,
        2);
    */
    LOG(INFO) << "clas shape " << hotword_size << " " << max_hotword_len << std::endl;
    
    /*
    std::vector<Ort::Value> input_onnx;
    input_onnx.emplace_back(std::move(onnx_hotword));
    */

    std::vector<std::vector<float>> result;
    try {
        //bmrt
        net_names = NULL;
        bmrt_get_network_names(p_bmrt_hw, &net_names);
        net_info = bmrt_get_network_info(p_bmrt_hw, net_names[0]);
        assert(NULL != net_info);
        bm_tensor_t input_tensors_hw[1];
        input_tensors_hw[0].shape = {2, {hotword_size, max_hotword_len}};
        input_tensors_hw[0].dtype = net_info->input_dtypes[0];
        if (is_1688) {
            size_t size = bmrt_tensor_bytesize(&input_tensors_hw[0]);
            bm_malloc_device_byte(bm_handle, &input_tensors_hw[0].device_mem, size);
        } else
            input_tensors_hw[0].device_mem = net_info->stages[0].input_mems[0];
        input_tensors_hw[0].st_mode = BM_STORE_1N;
        status = bm_memcpy_s2d_partial(bm_handle, input_tensors_hw[0].device_mem, (int32_t*)hotword_matrix.data(), hotword_size*max_hotword_len*sizeof(int32_t));
        assert(BM_SUCCESS == status);

        bm_tensor_t output_tensors_hw[1];
        if (is_1688) {
            bm_malloc_device_byte(bm_handle, &output_tensors_hw[0].device_mem, net_info->max_output_bytes[0]);
        } else
            output_tensors_hw[0].device_mem = net_info->stages[0].output_mems[0];

        //forward
        ret = bmrt_launch_tensor_ex(p_bmrt_hw, net_names[0], input_tensors_hw, 1, output_tensors_hw, 1, true, false);
        assert(true == ret);
        bm_thread_sync(bm_handle);
        /*
        auto outputTensor = hw_m_session->Run(Ort::RunOptions{nullptr}, hw_m_szInputNames.data(), input_onnx.data(), input_onnx.size(), hw_m_szOutputNames.data(), hw_m_szOutputNames.size());
        std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();

        int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
        float* floatData = outputTensor[0].GetTensorMutableData<float>(); // shape [max_hotword_len, hotword_size, dim]
        // get embedding by real hotword length
        assert(outputShape[0] == max_hotword_len);
        assert(outputShape[1] == hotword_size);
        embedding_dim = outputShape[2];
        */
        auto hotword_out_shape = output_tensors_hw[0].shape;
        std::vector<float> floatData(bmrt_shape_count(&hotword_out_shape));
        status = bm_memcpy_d2s_partial(bm_handle, floatData.data(), output_tensors_hw[0].device_mem, bmrt_tensor_bytesize(&output_tensors_hw[0]));
        assert(BM_SUCCESS == status);

        assert(hotword_out_shape.dims[0] == max_hotword_len);
        assert(hotword_out_shape.dims[1] == hotword_size);
        embedding_dim = hotword_out_shape.dims[2];

        for (int j = 0; j < hotword_size; j++)
        {
            int start_pos = hotword_size * (lengths[j] - 1) * embedding_dim + j * embedding_dim;
            std::vector<float> embedding;
            embedding.insert(embedding.begin(), floatData.begin() + start_pos, floatData.begin() + start_pos + embedding_dim);
            result.push_back(embedding);
        }
        if (is_1688){
            for (int i = 0; i < net_info->input_num; ++i) {
                bm_free_device(bm_handle, input_tensors_hw[i].device_mem);
            }
            for (int i = 0; i < net_info->output_num; ++i) {
                bm_free_device(bm_handle, output_tensors_hw[i].device_mem);
            }
        }
    }
    catch (std::exception const &e)
    {
        LOG(ERROR)<<e.what();
    }
    //PrintMat(result, "clas_embedding_output");
    return result;
}

Vocab* Paraformer::GetVocab()
{
    return vocab;
}

Vocab* Paraformer::GetLmVocab()
{
    return lm_vocab;
}

PhoneSet* Paraformer::GetPhoneSet()
{
    return phone_set_;
}

string Paraformer::Rescoring()
{
    LOG(ERROR)<<"Not Imp!!!!!!";
    return "";
}
} // namespace funasr
