---
layout: post
title: "2. Training Custom BPE Tokeniser"
posted: May 28, 2024
categories: Super-Fast-LLM-Training
live: true
---
The core of deep learning, the algorithm that actually does the “learning”, is stochastic gradient descent (SGD). The same algorithm powers logistic regression to ChatGPT and every neural network in between. SGD in turn defines many aspects of the problem. One such constraint is that data needs to be represented as numbers. And that’s not enough - the numeric representation of each feature in input data should be ordinal in nature. The constraint is trivially satisfied for numeric features but things get complicated for categorical features. In categorical features, the feature can only take one of N possible values. The values may not have any order to them and just assigning a numeric identifier to every possible value does not work with SGD.

Data used for language models contain a single categorical feature - word. A word, in English, can take up to 273,000 values (total number of English words in Oxford dictionary). There is no ordering amongst word meanings. The meaning of word “Apple” has no ordinal relationship with the meaning of word “Orange”. 

Therefore, instead of using a single categorical feature, we represent a word with 273,000 binary features. Each feature corresponds to presence of a word. Since a word consists of only one word (duh), only 1 out of those 273,000 features has value 1. Rest have value 0. Now each feature is ordinal in nature - 1 represents presence and 0 represents absence. Such an encoding scheme is called one-hot encoding. With one-hot encoding, each word in the dictionary is assigned a id between 0 to 272,999. Then, given a word, we can obtain its one-hot encoding by creating a vector of zeros of size 273,000 and setting the value at index corresponding to the word’s id as 1.

The process of assigning ids to words is called word-level tokenisation. The combination of tokenisation and one-hot encoding allows us to convert text into a format that SGD can work with. In this case, out text will be a sequence of one-hot vectors - each 273,000 elements in size.

## Types of Tokenisation

Word-level tokenisation is not the only way to convert text into an appropriate numeric representation. For instance, you could use ASCII encoding which assigns unique ids between 0 to 127 to every common English character and punctuations. Such an encoding would convert the text into a sequence of vectors - each 128 elements in size. The sequence will be longer this time since we have one vector for every character.

Word-level tokenisation results in sequence of extremely large vectors - even if we restrict to just English. For example, our small mono-lingual dataset, which has ten million total words, contains 353,515 unique words (much larger than dictionary size since variations of the same word like `dont` `Dont` `don’t` `Don’t` are all counted as a separate word). Which means, each word would have to be represented as a 353,515 sized vector! Additionally, the word `don’t` is a combination of two words `do` and `not` (which will also be present in the dataset somewhere else). It would be better to split words like `don’t` in the dataset to avoid counting such words as unique.

There are many libraries that use rules of grammar to perform such tokenization. [spaCy](https://spacy.io/) is one such popular library. The set of grammar rules used for tokenization is also called a language model (not to be confused with the language models we are building). The `en_core_web_sm` language model from spaCy library encodes the word `don’t` as two tokens `do` and `n’t` and the word `dont` as `do` and `nt`. 

```python
import spacy
nlp = spacy.load("en_core_web_sm")
tokens = nlp("don't dont")
print(list(tokens))
# ["do", "n't", "do", "nt"]
```

Using spaCy as our tokenizer, we get a vocabulary of 74,167 tokens. In comparison, the GPT-3.5-turbo model (LLM used for ChatGPT), whose training data included many languages (human and programming languages), has a vocabulary size of a hundred thousand tokens. 

The above number was obtained without normalizing the capitalization of words. If our entire text is converted to lowercase, we get 60,554 tokens. But converting entire text to lower case has its own problem - the model outputs will also be lowercase. The model will not learn (or even be able to predict) capitalization since capital letters are not part of the vocabulary. In some cases, it does not matter. In some other cases, rules of grammar can be used to post-process output and adjust the casing. In many other cases (like programming), this approach of lowercasing all text will not work well.

As we shall see in later sections, the number of unique tokens (also called size of vocabulary) dictates the size of embedding matrix and the final dense projection layer of the model. Therefore, it is important for it to be as small as possible. 

The other extreme tokenization strategy we discussed is to assign a unique id to every character in the dataset. Since our dataset is English based and does not include any complex characters/emojis, we can even use the ASCII encoding scheme. That way, we get a vocabulary size of just 128 (and some additional tokens for special markers like start and end of generation). But to support a full range of characters from all languages as well as emojis, we would need the Unicode encoding scheme which has a vocabulary size of 149,186. Moreover, encoding individual characters results in sub-optimal learning likely because of lack of context. Computers, which can also operate only on numbers, use such character level encodings for storing and operating on text.

Therefore, most LLMs do not use rule based or character based tokenizers. Instead, they use algorithms that find an encoding scheme that breaks words into smaller sub-word tokens. Such tokens may or may not have any inherent meaning (may or may not be present in dictionary) but they strike the right balance between providing enough context and reducing vocabulary size. Byte-level Byte Pair Encoding (BBPE) is one such popular tokenizer algorithm. It was initially designed as a text compression algorithm and was later adapted by OpenAI as a tokenizer for the GPT family of LLMs. Other popular tokenizer algorithms are WordPiece, SentencePiece and Unigram.

Before we dive into Byte Pair Encoding (BPE), we need to first understand how computers represent text as numbers.

## Unicode and it’s UTF-8 Representation

In UTF-8, each character (of about a million characters defined in Unicode standard) can be represented using one to four bytes. Most common english characters (all characters supported by ASCII encoding scheme) can be represented with just one byte.

For example, the text `Hello` when encoded using UTF-8, results in a array of five bytes - each character being represented with 1 byte. 1 byte can represent up to 256 unique characters. UTF-8 encodes the most common 128 characters using just 7 bits (the remaining bit is reserved to signal the presence of a multi-byte character). Rest of the characters require multiple bytes. For example, the string `Hello 😁` is encoded as `[b'H', b'e', b'l', b'l', b'o', b' ', b'\xf0', b'\x9f', b'\x98', b'\x81']`  - a total of 10 bytes. The emoji requires four bytes and during decoding, the byte `b'\xf0'` signals that the upcoming four bytes (including itself) should be decoded as a single character.

```python
print([bytes([b]) for b in "Hello 😁".encode("utf-8")])
# [b'H', b'e', b'l', b'l', b'o', b' ', b'\xf0', b'\x9f', b'\x98', b'\x81']
```

## Byte Pair Encoding (BPE)

The objective of BPE is to assign unique token to most common sequence of bytes in text. To achieve that, it uses the following algorithm -  

1. Assign ids 0-255 to all possible values that can be represented with a single byte.
2. Split text into a sequence of words based on a delimiter (most commonly “ “).
3. Encode each word in the text into a sequence of bytes with UTF-8 (a variable length encoding scheme). 
4. Maintain a count of occurrence of each pair of adjacent bytes (first and second, second and third, ..., second to last and last byte pairs) across every word of the text.
5. Assign a new id to the byte pair that occurs most frequently. Replace all occurrences of that pair in the text with the combined byte sequence.
6. Repeat till you reach a desired vocabulary size.

If you repreat the loop M times, you get a vocabulary size of M + 256.

The time complexity of this algorithm is O(M*N) where M is the number of times we run this loop and N is the number of byte pairs (actually it is better than that since number of byte pairs reduce at every iteration but we don't need to dive into the details to guess that the algorithm will take a lot of time on our dataset). Moreover, we will need a streaming version of the algorithm that does not require loading all the text into main memory. 

To address efficiency, we use a modified version of the above pseudocode. The algorithm first creates a map of all unique words to their UTF-8 encoding and count as shown below

```python
import glob
import json

word_counts_and_encodings = {}
total_word_count = 0

# Iterate over all jsonl files in the curated dataset
for file in glob.glob(
    "~/EduLLM/data/food-com-cc-cleaned/*.jsonl"
):
    with open(file, "r") as f:
        for line in f:
		        # Decode each line as JSON and get the text field
            text = json.loads(line.strip())["text"]
            words = text.split(" ")
            for word in words:
		            # For each word, store its count and utf-8 byte string
                if len(word) > 0:
                    total_word_count += 1
                    if word in word_counts_and_encodings:
                        word_counts_and_encodings[word]["count"] += 1
                    else:
                        word_counts_and_encodings[word] = {}
                        word_counts_and_encodings[word]["encoding"] = [
                            bytes([b]) for b in word.encode("utf-8")
                        ]
                        word_counts_and_encodings[word]["count"] = 1

print(f"Unique word count: {len(word_counts_and_encodings)}")
# Unique word count: 353515
print(f"Total word count: {total_word_count}")
# Total word count: 10053688
```

  

Then the BPE algorithm looks like the following -

```python
MAX_VOCABULARY_SIZE = 1024
vocabulary = {}
for i in range(256):
    vocabulary[bytes([i])] = i

while len(vocabulary) < MAX_VOCABULARY_SIZE:
    byte_pair_counts = {}
    most_frequent_pair = None
    most_frequent_count = 0
    for word, count_and_encoding in word_counts_and_encodings.items():
        byte_sequence = count_and_encoding["encoding"]
        if len(byte_sequence) > 1:
            for b1, b2 in zip(byte_sequence[:-1], byte_sequence[1:]):
                if (b1, b2) in byte_pair_counts:
                    byte_pair_counts[(b1, b2)] += count_and_encoding["count"]
                else:
                    byte_pair_counts[(b1, b2)] = count_and_encoding["count"]
                if byte_pair_counts[(b1, b2)] > most_frequent_count:
                    most_frequent_count = byte_pair_counts[(b1, b2)]
                    most_frequent_pair = (b1, b2)
    try:
        print(
            f"""Found pair {most_frequent_pair} 
                (decoded as {(most_frequent_pair[0] + most_frequent_pair[1]).decode('utf-8')}) 
                {most_frequent_count} times. Adding to vocabulary"""
        )
    except:
        print(
            f"Found pair {most_frequent_pair} (failed to decode) {most_frequent_count} times. Still adding to vocabulary"
        )
    vocabulary[most_frequent_pair[0] + most_frequent_pair[1]] = len(vocabulary)
    for word, count_and_encoding in word_counts_and_encodings.items():
        byte_sequence = count_and_encoding["encoding"]
        new_byte_sequence = []
        i = 0
        if len(byte_sequence) > 1:
            while i < len(byte_sequence) - 1:
                if (
                    most_frequent_pair[0] == byte_sequence[i]
                    and most_frequent_pair[1] == byte_sequence[i + 1]
                ):
                    new_byte_sequence.append(byte_sequence[i] + byte_sequence[i + 1])
                    i += 2
                else:
                    new_byte_sequence.append(byte_sequence[i])
                    i += 1
        if i == len(byte_sequence) - 1:
            new_byte_sequence.append(byte_sequence[i])
        count_and_encoding["encoding"] = new_byte_sequence
```

The resulting vocabulary contains 1,024 most common byte sequences. Running time of this BPE algorithm will scale linearly with number of merges (which is size of vocabulary - 256). On our dataset of 1 million words, it takes almost an hour for 768 merges (1024 - 256 = 768). 

These are some of the longest byte sequences (decoded as strings using UTF-8 and Unicode) present in the vocabulary

The top few are - `.photobucket.com/albums/`, `\nAdvertisement`, `ellipsis-horizontal\n`. These signal that we need to refine our data cleaning procedure to remove such examples.

We can also see that, in some cases, different versions of the same word are encoded as different tokens - `ingredients`, `\n\nIngredients:`, `Ingredients:`, `Ingredient`, `ingredient` , `redient` are all part of vocabulary.

We also observe tokens for commonly used words. For example, we see,

- Ingredients names like - `chicken`, `cheese`, `pepper`, `butter`, `vinegar`, `vanilla`, `mushroom`, `cinnamon`, `chocolate`
- Measurements like - `teaspoon`, `minutes`, `degrees`, `ounces`
- Mediums like - `bucket`, `skillet`, `cup`
- Verbs like - `season`, `simmer`, `bake`, `gather`, `ground`, `combine`

Finally, we see many sub-words that are not present in the dictionary but occur commonly in our text.

The results show how BPE efficiently assigns ids to most frequent sub-words (or complete words). The vocabulary for GPT do not contain such sub-words since the text used there had different characteristics. Therefore, if you are training a language model on a custom domain dataset (for example training a LM on a new language dataset), you could benefit greatly by training your own BPE vocabulary.