def load_encoder_decoder(input_shape, encoder_func,  encoder_name,
                         decoder_func, decoder_name):
    encoder = encoder_func(input_shape)
    decoder = decoder_func(encoder.get_output_shape_at(-1)[1:])

    _, encoder_weight_file = _get_model_files(encoder_name)
    _, decoder_weight_file = _get_model_files(decoder_name)

    encoder.load_weights(encoder_weight_file)
    decoder.load_weights(decoder_weight_file)
    print('Loaded model from disk')
    return encoder, decoder


def load_coder(input_shape, coder_func,  coder_name):
    coder = coder_func(input_shape)
    _, coder_weight_file = _get_model_files(coder_name)
    coder.load_weights(coder_weight_file)
    print 'Loaded %s from disk' % coder_name
    return coder


def save_model(model, model_name):
    model_file, weight_file = _get_model_files(model_name)
    # model_json = model.to_json()
    # with open(model_file, 'w') as json_file:
    #     json_file.write(model_json)
    model.save_weights(weight_file)
    print('Saved model to disk')


def _get_model_files(model_name):
    model_file = '%s.json' % model_name
    weight_file = '%s.h5' % model_name
    return model_file, weight_file
