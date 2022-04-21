import re
import emoji
import numpy as np
import os
import torch
import sys 
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def check_dir_exists(dir):
    if not os.path.exists(dir):
        sys.exit(0)

def save_checkpoint(state, checkpoint, name='last.pth.tar'):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'.
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, name)
    if not os.path.exists(checkpoint):
        #print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)

    torch.save(state, filepath)


def write_embeddings(embeddings, path):
    """ Write embeddings in a text file given by {path}.
    Args:
        path: (string) text file path 
        embeddings: (dict) contains users as keys and their learned embeddings as values
    """
    with open(path, 'w') as f:
        for user, embedding in embeddings.items():
                temp = str(user)

                for val in embedding:
                    temp += ' ' + str(round(val.item(), 4))

                f.write(temp + '\n')

def get_embeddings_dict_from_path(path):
    if not os.path.exists(path):
        print("{} does not exist !".format(path))
        return

    embeddings = {}
    
    counter = 0
    with open(path, 'r') as f:
        for line in f:
            counter += 1
            temp = line.strip().split(' ')
            user = temp[0]
            embeddings[user] = np.array(temp[1:], dtype=np.double)
    
    return embeddings


EMOJI_DESCRIPTION_SCRUB = re.compile(r':(\S+?):')
HASHTAG_BEFORE = re.compile(r'#(\S+)')
BAD_HASHTAG_LOGIC = re.compile(r'(\S+)!!')
FIND_MENTIONS = re.compile(r'@(\S+)')
LEADING_NAMES = re.compile(r'^\s*((?:@\S+\s*)+)')
TAIL_NAMES = re.compile(r'\s*((?:@\S+\s*)+)$')

def process_tweet(s, save_text_formatting=True, keep_emoji=False, keep_usernames=False):
    if save_text_formatting:
        s = re.sub(r'https\S+', r'', str(s))
        s = re.sub(r'http\S+', r'', str(s))
    else:
        s = re.sub(r'http\S+', r'', str(s))
        s = re.sub(r'https\S+', r' ', str(s))
        s = re.sub(r'x{3,5}', r' ', str(s))
    
    s = re.sub(r'\\n', ' ', s)
    s = re.sub(r'\s', ' ', s)
    s = re.sub(r'<br>', ' ', s)
    s = re.sub(r'&amp;', '&', s)
    s = re.sub(r'&#039;', "'", s)
    s = re.sub(r'&gt;', '>', s)
    s = re.sub(r'&lt;', '<', s)
    s = re.sub(r'\'', "'", s)

    if save_text_formatting:
        s = emoji.demojize(s)
    elif keep_emoji:
        s = emoji.demojize(s)
        s = s.replace('face_with', '')
        s = s.replace('face_', '')
        s = s.replace('_face', '')
        s = re.sub(EMOJI_DESCRIPTION_SCRUB, r' \1 ', s)
        s = s.replace('(_', '(')
        s = s.replace('_', ' ')

    s = re.sub(r"\\x[0-9a-z]{2,3,4}", "", s)
    
    if save_text_formatting:
        s = re.sub(HASHTAG_BEFORE, r'\1!!', s)
    else:
        s = re.sub(HASHTAG_BEFORE, r'\1', s)
        s = re.sub(BAD_HASHTAG_LOGIC, r'\1', s)
    
    if save_text_formatting:
        #@TODO 
        pass
    else:
        # If removing formatting, either remove all mentions, or just the @ sign.
        if keep_usernames:
            s = ' '.join(s.split())

            s = re.sub(LEADING_NAMES, r' ', s)
            s = re.sub(TAIL_NAMES, r' ', s)

            s = re.sub(FIND_MENTIONS, r'\1', s)
        else:
            s = re.sub(FIND_MENTIONS, r' ', s)
    #s = re.sub(re.compile(r'@(\S+)'), r'@', s)
    user_regex = r".?@.+?( |$)|<@mention>"    
    s = re.sub(user_regex," @user ", s, flags=re.I)
    
    # Just in case -- remove any non-ASCII and unprintable characters, apart from whitespace  
    s = "".join(x for x in s if (x.isspace() or (31 < ord(x) < 127)))
    s = ' '.join(s.split())
    return s

