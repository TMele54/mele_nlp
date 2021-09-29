from nlp import get_reviews,remove_special_char,remove_numbers,lower_case,tokenize,part_of_speech,bag_of_words,preserve_apostrohpy
import nltk

'''
    1. 	CC 	Coordinating conjunction
    2. 	CD 	Cardinal number
    3. 	DT 	Determiner
    4. 	EX 	Existential there
    5. 	FW 	Foreign word
    6. 	IN 	Preposition or subordinating conjunction
    7. 	JJ 	Adjective
    8. 	JJR 	Adjective, comparative
    9. 	JJS 	Adjective, superlative
    10. LS 	List item marker
    11. MD 	Modal
    12. NN 	Noun, singular or mass
    13. NNS 	Noun, plural
    14. NNP 	Proper noun, singular
    15. NNPS 	Proper noun, plural
    16. PDT 	Predeterminer
    17. POS 	Possessive ending
    18. PRP 	Personal pronoun
    19. PRP$ 	Possessive pronoun
    20. RB 	Adverb
    21. RBR 	Adverb, comparative
    22. RBS 	Adverb, superlative
    23. RP 	Particle
    24. SYM 	Symbol
    25. TO 	to
    26. UH 	Interjection
    27. VB 	Verb, base form
    28. VBD 	Verb, past tense
    29. VBG 	Verb, gerund or present participle
    30. VBN 	Verb, past participle
    31. VBP 	Verb, non-3rd person singular present
    32. VBZ 	Verb, 3rd person singular present
    33. WDT 	Wh-determiner
    34. WP 	Wh-pronoun
    35. WP$ 	Possessive wh-pronoun
    36. WRB 	Wh-adverb

'''

def generate_custom_grammar(part_of_speech_array):
    part_of_speech_array = set(bag_of_words(part_of_speech_array))
    types = {
        "PRP$":"",
        "VBG":"",
        "FW":"",
        "VBN":"",
        "''":"",
        "VBP":"",
        "WDT":"",
        "JJ":"",
        "WP":"",
        "VBZ":"",
        "DT":"",
        "RP":"",
        "$":"",
        "NN":"",
        "VBD":"",
        "POS":"",
        "TO":"",
        "PRP":"",
        "RB":"",
        "NNS":"",
        "NNP":"",
        "VB":"",
        "WRB":"",
        "CC":"",
        "PDT":"",
        "RBS":"",
        "RBR":"",
        "CD":"",
        "EX":"",
        "IN":"",
        "WP$":"",
        "MD":"",
        "NNPS":"",
        "JJS":"",
        "JJR":"",
        "SYM":"",
        "UH":"",
    }
    gammy = '''S -> NP VP
               VP -> VB NP | V NP PP
               PP -> P NP
               NP -> Det N | Det N PP
               N -> 'NN' | 'NNS' | 'NNP'
               Det -> 'DT'
               Adj -> 'JJ'
               V -> 'VBZ'| 'VB'
               P -> 'PP'
            '''

    print "concatenating tokens"
    for ay in part_of_speech_array:
        if "$" in ay[1] or "" in ay[1]:
            pass
        else:
            types[ay[1]] += '"'+ay[0]+'" | '

    print "expanding grammar"
    for type in types:
        if "$" in type:
            pass
        else:
            types[type] = types[type][:-2]
            types[type] += "\n"
            if types[type] != '\n':
                gammy += type+" -> "+types[type]
            else:
                pass

    return gammy

'''
    def gramify(poss):
        outer = []
        inner = []
        for i in range(0,len(poss)):
            length = len(poss[i])
            for j in range(0,length):
                poss[i][j] = list(poss[i][j])

                if poss[i][j][1] == 'NN':
                    poss[i][j][1] = 'N'
                elif poss[i][j][1] == 'VB':
                    poss[i][j][1] = 'V'
                elif poss[i][j][1]  == "DT":
                    poss[i][j][1] = 'Det'
                elif poss[i][j][1] == 'JJ':
                    poss[i][j][1] = 'Adj'
                elif poss[i][j][1] == 'PP':
                    poss[i][j][1] = 'P'
                else:
                    poss[i][j][1] = poss[i][j][1]

                inner.append(tuple(poss[i][j]))

            outer.append(inner)
            inner = []
        return outer
'''

data0 = get_reviews("../../data/movie_reviews/pos/")
data0 = tokenize(lower_case(remove_special_char(data0)))
data0 = preserve_apostrohpy(remove_numbers(data0))

#dataE = tokenize(lower_case(remove_special_char(data0)))
data = part_of_speech(data0)

#data = gramify(data)
grams = generate_custom_grammar(data)

print grams

grammar = nltk.CFG.fromstring(grams)
print grammar
rd_parser = nltk.RecursiveDescentParser(grammar)
print rd_parser
for tree in rd_parser.parse(data0[0]):
    print(tree)