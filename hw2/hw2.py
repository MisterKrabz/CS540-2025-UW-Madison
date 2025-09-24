import string
import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    # Implementing vectors e,s as lists (arrays) of length 26
    # with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)
    
def shred(filename):
    '''
    This function parses a .txt file into a dict of character counts, where
    the input file may contain any printable ASCII characters and can be short 
    (e.g. a single word) or long (e.g. an article). Ignores case, i.e. merges 
    'A' and 'a' counts together, and so on (this is known as case-folding). 
    Only counts characters A to Z (after case-folding), ignoring all other 
    characters such as space, punctuations, etc.

    Sample Input/Output functionality:
    Input: "Hi! I'll go :-)"
    Output: {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0,
             'G': 1, 'H': 1, 'I': 2, 'J': 0, 'K': 0, 'L': 2,
             'M': 0, 'N': 0, 'O': 1, 'P': 0, 'Q': 0, 'R': 0,
             'S': 0, 'T': 0, 'U': 0, 'V': 0, 'W': 0, 'X': 0,
             'Y': 0, 'Z': 0}

    Returns: dict of character counts
    '''

    boc=dict()
    with open (filename,encoding='utf-8') as f:
        corpus=f.read()
    f.close()
    #convert all lowercase alphabets to uppercase
    corpus=corpus.upper()
    #initialize X
    for a in string.ascii_uppercase:
        boc[a]=0
    for c in corpus:
        if c in string.ascii_uppercase:
            boc[c]+=1
    return boc



# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!
def main(): 
    # Check for the correct number of CLA
    if(len(sys.argv) != 4):
        print("Usage: python3 hw2.py <letter_file> <english_prior> <spanish_prior>")
        sys.exit(1)
    
    # Parse command line arguments
    english_prior = float(sys.argv[2])
    spanish_prior = float(sys.argv[3])

    # Get the character counts from the input file
    character_counts = shred(sys.argv[1])

    # Get the language model parameter vectors
    e, s = get_parameter_vectors()

    # Q1 ([33] points): Compute X1 log e1 and X1 log s1 (remember that X1 is the
    # count of character ’A’). Print “Q1” then these values up to 4 decimal places
    # on two separate lines.
    print("Q1")
    # count of character 'A'
    x1 = character_counts['A']
    # probability of 'A' in english 
    e1 = e[0]
    # probability of 'A' in spanish 
    s1 = s[0]

    # calculate and print
    q1_english_val = x1 * math.log(e1) 
    q1_spanish_val = x1 * math.log(s1) 
    
    print(f"{q1_english_val:.4f}")
    print(f"{q1_spanish_val:.4f}")

    # Q2 ([33] points): Compute F (English) and F (Spanish). Similarly, print
    # “Q2” followed by their values up to 4 decimal places on two separate lines.
    print("Q2")
    # Initialize scores with the log of the prior probabilities
    f_english = math.log(english_prior)
    f_spanish = math.log(spanish_prior)

    # loop through the characters
    for i in range(26):
        char = chr(ord('A') + i)
        count = character_counts[char]

        # Add the term X_i * log(p_i) to the total log probability
        f_english += count * math.log(e[i])
        f_spanish += count * math.log(s[i])
    
    print(f"{f_english:.4f}")
    print(f"{f_spanish:.4f}")

    # Q3 ([34] points): Compute P (Y = English | X). Print “Q3” then this value
    # up to 4 decimal places
    print("Q3")
    prob_english = 0.0
    # Calculate the difference 
    f_diff = f_spanish - f_english

    # if/else check 
    if f_diff >= 100:
        prob_english = 0.0
    elif f_diff <= -100:
        prob_english = 1.0
    else: 
        prob_english = 1 / (1 + math.exp(f_diff))
    
    print(f"{prob_english:.4f}")

if __name__ == "__main__":
    main()