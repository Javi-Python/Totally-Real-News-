import tensorflow
tensorflow.__version__
import gpt_2_simple as gpt2
from datetime import datetime



def generatefakes(length, temperature, prefix, nsamples):
    '''
    length: Number of tokens to generate (default 1023, the maximum)
    temperature: The higher the temperature, the crazier the text (default 0.7, recommended to keep between 0.7 and 1.0)
    top_k: Limits the generated guesses to the top k guesses (default 0 which disables the behavior; if the generated output is super crazy, you may want to set top_k=40)
    top_p: Nucleus sampling: limits the generated guesses to a cumulative probability. (gets good results on a dataset with top_p=0.9)
    truncate: Truncates the input text until a given sequence, excluding that sequence (e.g. if truncate='<|endoftext|>', the returned text will include everything before the first <|endoftext|>). It may be useful to combine this with a smaller length if the input texts are short.
    include_prefix: If using truncate and include_prefix=False, the specified prefix will not be included in the returned text.
    nsamples: how many samples you want to generate
    prefix: How you want the sentence to start

    '''
    model_path = '../GPT2-Model-Fakes/checkpoint_run4.tar'
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name = 'run4', model_dir = '../GPT2-Model-Fakes', checkpoint_dir = '../GPT2-Model-Fakes/checkpoint 2')
    return gpt2.generate(sess,
                length=length,
                top_k = 0,
                temperature=temperature,
                prefix=prefix,
                nsamples=nsamples,
                batch_size=nsamples,
                run_name='run4',
                checkpoint_dir ='../GPT2-Model-Fakes/checkpoint 2', top_p = 0.95,
                return_as_list=True)