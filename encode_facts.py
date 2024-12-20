from sentence_transformers import SentenceTransformer
import pickle
import json

device = 'cuda'
model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
print("mdoel_sentenceTransformer")
with open('./counterfact.json', 'r') as f:
    lines = json.load(f)
print("datasize:",len(lines))
sentences = []
subjects = []
for i, line in enumerate(lines):

    # print(str(i) + "/" + str(len(lines)))
    new_fact = line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject']) + ' ' + line['requested_rewrite']['target_new']['str']
    target_new = line['requested_rewrite']['target_new']['str']
    target_true = line['requested_rewrite']['target_true']['str']
    paraphrases = line['paraphrase_prompts']
    neighbors = line['neighborhood_prompts']
    subject = line['requested_rewrite']['subject']
    # if i <= 2000:
    #     sentences.append(
    #         f"New Fact: {new_fact}\nPrompt: {line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject'])}")
    #     subjects.append(subject)
    #     for p in paraphrases:
    #         sentences.append(f"New Fact: {new_fact}\nPrompt: {p}")
    #         subjects.append(subject)
    #     for n in neighbors:
    #         sentences.append(f"New Fact: {new_fact}\nPrompt: {n}")
    #         subjects.append(subject)
    # else:
    #     sentences.append(f"New Fact: {new_fact}\nPrompt: {new_fact}")
    #     subjects.append(subject)
    #     for p in paraphrases:
    #         sentences.append(f"New Fact: {new_fact}\nPrompt: {p} {target_new}")
    #         subjects.append(subject)
    #     for n in neighbors:
    #         sentences.append(f"New Fact: {new_fact}\nPrompt: {n} {target_true}")
    #         subjects.append(subject)python
    if i > 10000:
        sentences.append(f"New Fact: {new_fact}\nPrompt: {new_fact}")
    else:
        sentences.append(f"New Fact: {new_fact}\nPrompt: {line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject'])}")
    subjects.append(subject)
    # for p in paraphrases:
    #     if i> 2000:
    #         sentences.append(f"New Fact: {new_fact}\nPrompt: {p} {target_new}")
    #     else:
    #         sentences.append(f"New Fact: {new_fact}\nPrompt: {p}")
    #     subjects.append(subject)
    # for n in neighbors:
    #     if i > 2000:
    #         sentences.append(f"New Fact: {new_fact}\nPrompt: {n} {target_true}")
    #     else:
    #         sentences.append(f"New Fact: {new_fact}\nPrompt: {n}")
    #     subjects.append(subject)
# sentences = [line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject']) + ' ' + line['requested_rewrite']['target_new']['str'] for line in lines]
# sentences = ['This framework generates embeddings for each input sentence',
#     'Sentences are passed as a list of string.',
#     'The quick brown fox jumps over the lazy dog.']

print("embedding start")
embeddings = model.encode(sentences)
print("embedding end")
print("writing to pkl")

#Store sentences & embeddings on disc
with open('embeddings_ori.pkl', "wb") as fOut:
    pickle.dump({'sentences': sentences, 'embeddings': embeddings, 'subjects': subjects}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

#Load sentences & embeddings from disc
# with open('embeddings.pkl', "rb") as fIn:
#     stored_data = pickle.load(fIn)
#     stored_sentences = stored_data['sentences']
#     stored_embeddings = stored_data['embeddings']