import torch
import numpy as np

def session_maxlen(session_last_dict):
    leng = []
    for user in session_last_dict.keys():
        a = session_last_dict[user]
        part_max_len = max(len(lst) for lst in a)
        leng.append(part_max_len)
    max_len = max(leng)
    return max_len

def flatten(lst):
    flat_list = []
    for sublist in lst:
        if len(lst) == 1:
            # flat_list.append(sublist)
            flat_list = sublist
        else:
            flat_list += sublist
    return flat_list
def remove_extra_brackets(lst):
    if isinstance(lst, list) and len(lst) == 1:
        return remove_extra_brackets(lst[0])
    elif isinstance(lst, list):
        return [remove_extra_brackets(sublist) for sublist in lst]
    else:
        return lst

def self_friend_neg2():
    aug_friend = torch.load("./data/new_datasets/nyc_data_judge_friend.pt")
    data = np.load('./data/new_datasets/train_augement_nyc.npz', allow_pickle=True)
    user_index = data['trainX_users']
    session_locations = data['trainX_session_locations']

    lastuser_dict = {user: [] for user in user_index}

    for i in range(len(user_index) - 1):
        if user_index[i] != user_index[i + 1]:
            lastuser_dict[user_index[i]].append(i)
        lastuser_dict[user_index[i + 1]].append(i + 1)

    session_last_dict = {user: [flatten(session_locations[lastuser_dict[user]])]
                         for user in lastuser_dict.keys()}

    dl_self = {}
    dl_friend = {}
    dl_neg = {}

    for f_key, f_value in aug_friend.items():
        for session_key, session_value in session_last_dict.items():
            if session_key == f_key:
                dl_self.setdefault(f_key, []).extend(session_value)
            elif session_key == f_value[0]:
                dl_friend.setdefault(f_value[0], []).extend(session_value)
            elif session_key == f_value[1]:
                dl_neg.setdefault(f_value[1], []).extend(session_value)

    for dictionary in [dl_self, dl_friend, dl_neg]:
        for key in dictionary:
            dictionary[key] = dictionary[key][0]

    max_len_all = max(session_maxlen(dictionary) for dictionary in [dl_self, dl_friend, dl_neg])

    return dl_self, dl_friend, dl_neg, max_len_all
def dict_convert(data):
    new_dict = {'seq':[], 'len':[], 'user':[]}
    for key, value in data.items():
        for value_num in range(len(value)):
            if isinstance(value[value_num], list):
                new_dict['user'].append(key)
                new_dict['seq'].append(value[value_num])
                # print(key)
                # print(value[value_num])
                new_dict['len'].append(len(value[value_num]))
            else:
                continue
    return new_dict

def dict_seq_equal_padding(dict, max_len):
    dict_new = {}
    for key in dict.keys():
        dict_new[key] = []
    for keys, values in dict.items():
        a_temp = values
        a_padded = [[0] * (max_len - len(lst)) + lst for lst in a_temp]
        dict_new[keys].append(a_padded)
    for new_key, new_value in dict_new.items():
        dict_new_clean = remove_extra_brackets(new_value)
        dict_new[new_key] = dict_new_clean
    return dict_new


# if __name__ == '__main__':
#     dl_self, dl_friend, dl_neg, max_len_all = self_friend_neg2()
#     print(dl_self, dl_friend, dl_neg, max_len_all)
