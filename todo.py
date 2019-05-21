import torch
import torch.nn.functional as F
from config import config

_config = config()

print('asdasd')
def evaluate(golden_list, predict_list):
    num_list_1 = len(golden_list)
    num_list_2 = len(golden_list[0])
    num_gt = 0
    num_pre = 0
    
    for i in range(num_list_1):
        for j in range(len(predict_list[i])):
            if golden_list[i][j][0] == 'B':
                num_gt += 1
            if predict_list[i][j][0] == 'B':
                num_pre += 1
                    
    tp = 0
    for i in golden_list:
        i += ['$']
    
    for i in range(num_list_1):
        B_index = 0
        end_index = 0
        current_index = 0
        while current_index < len(golden_list[i]):
            if golden_list[i][current_index][0] == 'B':
                B_index = current_index
                end_index = B_index + 1
                while golden_list[i][end_index][0] == 'I':
                    end_index += 1
                if golden_list[i][B_index:end_index] == predict_list[i][B_index:end_index]:
                    if current_index == len(predict_list[i]) - 1:
                        tp += 1
                    elif predict_list[i][end_index][0] != 'I':
                        tp += 1
                current_index = end_index
            else:
                current_index += 1
    if tp == 0:
        return 0
    P = tp/num_gt
    R = tp/num_pre
    F1 = ( 2 * P * R ) / (P + R)

    
    return F1

            
       
def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate     = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate   = F.tanh(cellgate)
    outgate    = F.sigmoid(outgate)
    cy = (forgetgate * cx) + ( (1 - forgetgate) * cellgate)
    hy = outgate * F.tanh(cy)
    return hy, cy


##n = [['B-TAR','I-TAR','O','B-HYP','O']]
##o = [['B-TAR','O','O','B-HYP','I-HYP']]
##
##p = [['B-TAR','B-TAR','B-TAR','I-TAR']]
##q = [['B-TAR','B-TAR','B-TAR','O']]
##
##aa = [['B-TAR','I-TAR','O','B-HYP'],['B-TAR','O','O','B-HYP']]
##bb = [['O','O','B-HYP','I-HYP'],['O','O','O','O']]
##
##cc = [['B-TAR','O', 'I-TAR', 'O','B-TAR'],['B-TAR','B-TAR','I-TAR','O']]
##dd = [['B-TAR','O', 'O', 'O','O'],['O','O','O','O']]
##
##
##print(evaluate(n,o))
##print(evaluate(p,q))
##print(evaluate(aa,bb))
##print(evaluate(cc,dd))






def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    x,y,z = batch_char_index_matrices.size()
    
    c = batch_char_index_matrices.view(-1, z)
    batch_word_len_lists = batch_word_len_lists.view(-1)
    input_char_embeds = model.char_embeds(c)

    perm_idx, sorted_batch_word_len_lists = model.sort_input(batch_word_len_lists)    
    input_char_embeds = input_char_embeds[perm_idx]
#     print('#### reorder: ', input_char_embeds.size())

    _, desorted_indices = torch.sort(perm_idx, descending=False)

    output_sequence = pack_padded_sequence(input_char_embeds, sorted_batch_word_len_lists.data.tolist(), batch_first=True)
#     print('after padded: ',output_sequence.data.size())
    output_sequence, (h1, h2)  = model.char_lstm(output_sequence)
#     print('%%%%%%%:', output_sequence.data.size())
#     print('After lstm. It should be 2,14,50 :', h1.size(), h2.size())
#     print(h1.data)
#     print(h2.data)
#     print(h1.data[1])
    h1[0] = h1[0][desorted_indices]
#     _, desorted_indices = torch.sort(desorted_indices, descending=True)
    h1[1] = h1[1][desorted_indices]
#     print(h1.data[0])
#     print(h1.data[1])
    print('After packed :',  h1.data.size())
    new = torch.cat((h1[0],h1[1]),dim = -1)
    new = new.view(x,y,100)
    return new
    # Get hidden state of the shape [2,14,50].
    # Recover the hidden_states corresponding to the sorted index.
    # Re-shape it to get a Tensor the shape [2,7,100].
    #return result

