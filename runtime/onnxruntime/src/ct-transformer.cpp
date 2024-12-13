/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#include "precomp.h"

namespace funasr {
CTTransformer::CTTransformer()
:env_(ORT_LOGGING_LEVEL_ERROR, ""),session_options{}
{
}

void CTTransformer::InitPunc(const std::string &punc_model, const std::string &punc_config, const std::string &token_file, int thread_num){
    /*
    session_options.SetIntraOpNumThreads(thread_num);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_options.DisableCpuMemArena();
    */

    try{
        //m_session = std::make_unique<Ort::Session>(env_, ORTSTRING(punc_model).c_str(), session_options);
        status = bm_dev_request(&bm_handle, DEV_ID);
        assert(BM_SUCCESS == status);

        p_bmrt = bmrt_create(bm_handle);
        assert(NULL != p_bmrt);
        ret = bmrt_load_bmodel(p_bmrt, punc_model.c_str());
        assert(true == ret);

        net_names = NULL;
        bmrt_get_network_names(p_bmrt, &net_names);
        net_info = bmrt_get_network_info(p_bmrt, net_names[0]);
        assert(NULL != net_info);
        LOG(INFO) << "Successfully load model from " << punc_model;
    }
    catch (std::exception const &e) {
        //LOG(ERROR) << "Error when load punc onnx model: " << e.what();
        LOG(ERROR) << "Error when load punc bmodel: " << e.what();
        exit(-1);
    }
    // read inputnames outputnames
    //GetInputNames(m_session.get(), m_strInputNames, m_szInputNames);
    //GetOutputNames(m_session.get(), m_strOutputNames, m_szOutputNames);
    
	m_tokenizer.OpenYaml(punc_config.c_str(), token_file.c_str());
    m_tokenizer.JiebaInit(punc_config);
}

CTTransformer::~CTTransformer()
{
    if(p_bmrt){
        bmrt_destroy(p_bmrt);
    }
    if(bm_handle){
        bm_dev_free(bm_handle);
    }
}

string CTTransformer::AddPunc(const char* sz_input, std::string language)
{
    string strResult;
    vector<string> strOut;
    vector<int> InputData;
    m_tokenizer.Tokenize(sz_input, strOut, InputData); 

    int nTotalBatch = ceil((float)InputData.size() / TOKEN_LEN);
    int nCurBatch = -1;
    int nSentEnd = -1, nLastCommaIndex = -1;
    vector<int32_t> RemainIDs; // 
    vector<string> RemainStr; //
    vector<int> NewPunctuation; //
    vector<string> NewString; //
    vector<string> NewSentenceOut;
    vector<int> NewPuncOut;
    int nDiff = 0;
    for (size_t i = 0; i < InputData.size(); i += TOKEN_LEN)
    {
        nDiff = (i + TOKEN_LEN) < InputData.size() ? (0) : (i + TOKEN_LEN - InputData.size());
        vector<int32_t> InputIDs(InputData.begin() + i, InputData.begin() + i + (TOKEN_LEN - nDiff));
        vector<string> InputStr(strOut.begin() + i, strOut.begin() + i + (TOKEN_LEN - nDiff));
        InputIDs.insert(InputIDs.begin(), RemainIDs.begin(), RemainIDs.end()); // RemainIDs+InputIDs;
        InputStr.insert(InputStr.begin(), RemainStr.begin(), RemainStr.end()); // RemainStr+InputStr;

        auto Punction = Infer(InputIDs);
        nCurBatch = i / TOKEN_LEN;
        if (nCurBatch < nTotalBatch - 1) // not the last minisetence
        {
            nSentEnd = -1;
            nLastCommaIndex = -1;
            for (int nIndex = Punction.size() - 2; nIndex > 0; nIndex--)
            {
                if (m_tokenizer.Id2Punc(Punction[nIndex]) == m_tokenizer.Id2Punc(PERIOD_INDEX) || m_tokenizer.Id2Punc(Punction[nIndex]) == m_tokenizer.Id2Punc(QUESTION_INDEX))
                {
                    nSentEnd = nIndex;
                    break;
                }
                if (nLastCommaIndex < 0 && m_tokenizer.Id2Punc(Punction[nIndex]) == m_tokenizer.Id2Punc(COMMA_INDEX))
                {
                    nLastCommaIndex = nIndex;
                }
            }
            if (nSentEnd < 0 && InputStr.size() > CACHE_POP_TRIGGER_LIMIT && nLastCommaIndex > 0)
            {
                nSentEnd = nLastCommaIndex;
                Punction[nSentEnd] = PERIOD_INDEX;
            }
            RemainStr.assign(InputStr.begin() + (nSentEnd + 1), InputStr.end());
            RemainIDs.assign(InputIDs.begin() + (nSentEnd + 1), InputIDs.end());
            InputStr.assign(InputStr.begin(), InputStr.begin() + (nSentEnd + 1));  // minit_sentence
            Punction.assign(Punction.begin(), Punction.begin() + (nSentEnd + 1));
        }
        
        NewPunctuation.insert(NewPunctuation.end(), Punction.begin(), Punction.end());
        vector<string> WordWithPunc;
        for (int i = 0; i < InputStr.size(); i++)
        {
            // if (i > 0 && !(InputStr[i][0] & 0x80) && (i + 1) <InputStr.size() && !(InputStr[i+1][0] & 0x80))// �м��Ӣ�ģ�
            if (i > 0 && !(InputStr[i-1][0] & 0x80) && !(InputStr[i][0] & 0x80))
            {
                InputStr[i] = " " + InputStr[i];
            }
            WordWithPunc.push_back(InputStr[i]);

            if (Punction[i] != NOTPUNC_INDEX) // �»���
            {
                WordWithPunc.push_back(m_tokenizer.Id2Punc(Punction[i]));
            }
        }

        NewString.insert(NewString.end(), WordWithPunc.begin(), WordWithPunc.end()); // new_mini_sentence += "".join(words_with_punc)
        NewSentenceOut = NewString;
        NewPuncOut = NewPunctuation;
        // last mini sentence
        if(nCurBatch == nTotalBatch - 1)
        {
            if (NewString[NewString.size() - 1] == m_tokenizer.Id2Punc(COMMA_INDEX) || NewString[NewString.size() - 1] == m_tokenizer.Id2Punc(DUN_INDEX))
            {
                NewSentenceOut.assign(NewString.begin(), NewString.end() - 1);
                NewSentenceOut.push_back(m_tokenizer.Id2Punc(PERIOD_INDEX));
                NewPuncOut.assign(NewPunctuation.begin(), NewPunctuation.end() - 1);
                NewPuncOut.push_back(PERIOD_INDEX);
            }
            else if (NewString[NewString.size() - 1] != m_tokenizer.Id2Punc(PERIOD_INDEX) && NewString[NewString.size() - 1] != m_tokenizer.Id2Punc(QUESTION_INDEX))
            {
                NewSentenceOut = NewString;
                NewSentenceOut.push_back(m_tokenizer.Id2Punc(PERIOD_INDEX));
                NewPuncOut = NewPunctuation;
                NewPuncOut.push_back(PERIOD_INDEX);
            }
        }
    }

    for (auto& item : NewSentenceOut){
        strResult += item;
    }
    
    if(language == "en-bpe"){
        std::vector<std::string> chineseSymbols;
        chineseSymbols.push_back("，");
        chineseSymbols.push_back("。");
        chineseSymbols.push_back("、");
        chineseSymbols.push_back("？");

        std::string englishSymbols = ",.,?";
        for (size_t i = 0; i < chineseSymbols.size(); i++) {
            size_t pos = 0;
            while ((pos = strResult.find(chineseSymbols[i], pos)) != std::string::npos) {
                strResult.replace(pos, 3, 1, englishSymbols[i]);
                pos++;
            }
        }
    }

    return strResult;
}

vector<int> CTTransformer::Infer(vector<int32_t> input_data)
{
    //Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    vector<int> punction;
    /*
    std::array<int64_t, 2> input_shape_{ 1, (int64_t)input_data.size()};
    Ort::Value onnx_input = Ort::Value::CreateTensor<int32_t>(
        m_memoryInfo,
        input_data.data(),
        input_data.size(),
        input_shape_.data(),
        input_shape_.size());

    std::array<int32_t,1> text_lengths{ (int32_t)input_data.size() };
    std::array<int64_t,1> text_lengths_dim{ 1 };
    Ort::Value onnx_text_lengths = Ort::Value::CreateTensor(
        m_memoryInfo,
        text_lengths.data(),
        text_lengths.size() * sizeof(int32_t),
        text_lengths_dim.data(),
        text_lengths_dim.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    std::vector<Ort::Value> input_onnx;
    input_onnx.emplace_back(std::move(onnx_input));
    input_onnx.emplace_back(std::move(onnx_text_lengths));
    */
    // input tensor of punc
    bm_tensor_t input_tensors_punc[net_info->input_num];
    bmrt_tensor(&input_tensors_punc[0], p_bmrt, BM_INT32, {2, {1, (int)input_data.size()}});
    status = bm_memcpy_s2d_partial(bm_handle, input_tensors_punc[0].device_mem, input_data.data(), input_data.size()*sizeof(int32_t));
    assert(BM_SUCCESS == status);

    bmrt_tensor(&input_tensors_punc[1], p_bmrt, BM_INT32, {1, {1}});
    int32_t tl[1] = {static_cast<int32_t>(input_data.size())};
    status = bm_memcpy_s2d_partial(bm_handle, input_tensors_punc[1].device_mem, tl, sizeof(int32_t));
    assert(BM_SUCCESS == status);

    // output tensor of punc
    bm_tensor_t output_tensors_punc[net_info->output_num];
    status = bm_malloc_device_byte(bm_handle, &output_tensors_punc[0].device_mem, net_info->max_output_bytes[0]);
    assert(BM_SUCCESS == status);

        
    try {
        // forward
        ret = bmrt_launch_tensor_ex(p_bmrt, net_names[0], input_tensors_punc, 2, output_tensors_punc, 1, true, false);
        assert(true == ret);
        bm_thread_sync(bm_handle);
        /*
        auto outputTensor = m_session->Run(Ort::RunOptions{nullptr}, m_szInputNames.data(), input_onnx.data(), m_szInputNames.size(), m_szOutputNames.data(), m_szOutputNames.size());
        std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();

        int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
        float * floatData = outputTensor[0].GetTensorMutableData<float>();
        */
        auto punc_out_size = bmrt_tensor_bytesize(&output_tensors_punc[0]);
        auto punc_out_shape = output_tensors_punc[0].shape;
        auto outputCount = bmrt_shape_count(&punc_out_shape);
        float* floatData = new float[outputCount];
        status = bm_memcpy_d2s_partial(bm_handle, floatData, output_tensors_punc[0].device_mem, punc_out_size);
        assert(BM_SUCCESS == status);

        for (int i = 0; i < outputCount; i += CANDIDATE_NUM)
        {
            int index = Argmax(floatData + i, floatData + i + CANDIDATE_NUM-1);
            punction.push_back(index);
        }
        delete[] floatData;
        // free device memory
        for (int i = 0; i < net_info->output_num; ++i) {
            bm_free_device(bm_handle, input_tensors_punc[i].device_mem);
        }
        for (int i = 0; i < net_info->output_num; ++i) {
            bm_free_device(bm_handle, output_tensors_punc[i].device_mem);
        }
    }
    catch (std::exception const &e)
    {
        LOG(ERROR) << "Error when run punc onnx forword: " << (e.what());
    }
    return punction;
}

} // namespace funasr
