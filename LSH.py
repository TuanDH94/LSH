

from __future__ import division
import os
import re
import random
import time
import binascii
from bisect import bisect_right
from heapq import heappop, heappush

# This is the number of components in the resulting MinHash signatures.
# Correspondingly, it is also the number of random hash functions that
# we will need in order to calculate the MinHash.
import sys

numHashes = 200
numBands = 20
numRows = 10
threshold = 0.8
# You can run this code for different portions of the dataset.
# It ships with data set sizes 100, 1000, 2500, and 10000.
numDocs = -1
dataFile = "./data/real_data.csv"
truthFile = "./result/real_result"

# =============================================================================
#                  Parse The Ground Truth Tables
# =============================================================================
# Build a dictionary mapping the document IDs to their plagiaries, and vice-
# versa.
# Step 1: Doc du lieu tu file dang text
# Chuyen tu dinh dang document ban dau sang dinh dang k-shingle
# Ta chon k = 3
# Thuc hien hash cac k-shingle thanh cac so tu nhien trong khoang tu 0 - (2^32 - 1)
# Open the truth file.
f = open(truthFile, "rU")

print "Shingling articles..."

curShingleID = 0

docsAsShingleSets = {}

f = open(dataFile, "rU")

docNames = []

t0 = time.time()

totalShingles = 0

for line in f:
    numDocs += 1

    if(numDocs == 0):
        continue

    if(numDocs > 5000):
        break

    fields = line.split("\t")
    words = fields[3].split(" ")

    docID = fields[0]

    docNames.append(docID)

    shinglesInDoc = set()

    for index in range(0, len(words) - 2):

        shingle = words[index] + " " + words[index + 1] + " " + words[index + 2]

        crc = binascii.crc32(shingle) & 0xffffffff

        shinglesInDoc.add(crc)

    sorted(shinglesInDoc)

    docsAsShingleSets[docID] = shinglesInDoc


    totalShingles = totalShingles + (len(words) - 2)


f.close()

numDocs -= 1

print '\nShingling ' + str(numDocs) + ' docs took %.2f sec.' % (time.time() - t0)

print '\nAverage shingles per doc: %.2f' % (totalShingles / numDocs)


# =============================================================================
#                     Define Triangle Matrices
# =============================================================================

# Define virtual Triangle matrices to hold the similarity values. For storing
# similarities between pairs, we only need roughly half the elements of a full
# matrix. Using a triangle matrix requires less than half the memory of a full
# matrix, and can protect the programmer from inadvertently accessing one of
# the empty/invalid cells of a full matrix.

# Calculate the number of elements needed in our triangle matrix
numElems = int(numDocs * (numDocs - 1) / 2)

# Initialize two empty lists to store the similarity values.
# 'JSim' will be for the actual Jaccard Similarity values.
# 'estJSim' will be for the estimated Jaccard Similarities found by comparing
# the MinHash signatures.
JSim = [0 for x in range(numElems)]
estJSim = [0 for x in range(numElems)]


# Define a function to map a 2D matrix coordinate into a 1D index.
def getTriangleIndex(i, j):
    # If i == j that's an error.
    if i == j:
        sys.stderr.write("Can't access triangle matrix with i == j")
        sys.exit(1)
    # If j < i just swap the values.
    if j < i:
        temp = i
        i = j
        j = temp

    # Calculate the index within the triangular array.
    # This fancy indexing scheme is taken from pg. 211 of:
    # http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
    # But I adapted it for a 0-based index.
    # Note: The division by two should not truncate, it
    #       needs to be a float.
    k = int(i * (numDocs - (i + 1) / 2.0) + j - i) - 1

    return k




# =============================================================================
#                 Generate MinHash Signatures
# =============================================================================

# Time this step.
t0 = time.time()

print '\nGenerating random hash functions...'

# Record the maximum shingle ID that we assigned.
maxShingleID = 2 ** 32 - 1

# We need the next largest prime number above 'maxShingleID'.
# I looked this value up here:
# http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
nextPrime = 4294967311


# Our random hash function will take the form of:
#   h(x) = (a*x + b) % c
# Where 'x' is the input value, 'a' and 'b' are random coefficients, and 'c' is
# a prime number just greater than maxShingleID.

# Generate a list of 'k' random coefficients for the random hash functions,
# while ensuring that the same value does not appear multiple times in the
# list.
def pickRandomCoeffs(k):
    # Create a list of 'k' random values.
    randList = []

    while k > 0:
        # Get a random shingle ID.
        randIndex = random.randint(0, maxShingleID)

        # Ensure that each random number is unique.
        #while randIndex in randList:
         #   randIndex = random.randint(0, maxShingleID)

            # Add the random number to the list.
        randList.append(randIndex)
        k = k - 1

    return randList


# For each of the 'numHashes' hash functions, generate a different coefficient 'a' and 'b'.
coeffA = pickRandomCoeffs(numHashes)
coeffB = pickRandomCoeffs(numHashes)

print '\nGenerating MinHash signatures for all documents...'

# List of documents represented as signature vectors
signatures = []

# Rather than generating a random permutation of all possible shingles,
# we'll just hash the IDs of the shingles that are *actually in the document*,
# then take the lowest resulting hash code value. This corresponds to the index
# of the first shingle that you would have encountered in the random order.

# For each document...
for docID in docNames:

    # Get the shingle set for this document.
    shingleIDSet = docsAsShingleSets[docID]

    # The resulting minhash signature for this document.
    signature = []

    # For each of the random hash functions...
    for i in range(0, numHashes):

        # For each of the shingles actually in the document, calculate its hash code
        # using hash function 'i'.

        # Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
        # the maximum possible value output by the hash.
        minHashCode = nextPrime + 1

        # For each shingle in the document...
        for shingleID in shingleIDSet:
            # Evaluate the hash function.
            hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime

            # Track the lowest hash code seen.
            if hashCode < minHashCode:
                minHashCode = hashCode

        # Add the smallest hash code value as component number 'i' of the signature.
        signature.append(minHashCode)

    # Store the MinHash signature for this document.
    signatures.append(signature)

# Calculate the elapsed time (in seconds)
elapsed = (time.time() - t0)

print "\nGenerating MinHash signatures took %.2fsec" % elapsed

# =============================================================================
#                     Compare All Signatures
# =============================================================================

print '\nComparing all signatures...'

# Creates a N x N matrix initialized to 0.

# Time this step.
t0 = time.time()

# For each of the test documents...
for i in range(0, numDocs):
    # Get the MinHash signature for document i.
    signature1 = signatures[i]

    # For each of the other test documents...
    for j in range(i + 1, numDocs):

        # Get the MinHash signature for document j.
        signature2 = signatures[j]


        # Count the number of positions in the minhash signature which are equal.
        counts = 0
        for k in range(0, numBands):
            count = 0
            for l in range(0, numRows):
                count = count + (signature1[k * numRows + l] == signature2[k * numRows + l])
            if (count / numRows > threshold ):
                counts = count
                break
        # Record the percentage of positions which matched.
        estJSim[getTriangleIndex(i, j)] = ( counts / numRows)

# Calculate the elapsed time (in seconds)
elapsed = (time.time() - t0)

print "\nComparing MinHash signatures took %.2fsec" % elapsed

# =============================================================================
#                   Display Similar Document Pairs
# =============================================================================

# Count the true positives and false positives.
tp = 0
fp = 0

print "\nList of Document Pairs with J(d1,d2) more than", threshold
print "Values shown are the estimated Jaccard similarity and the actual"
print "Jaccard similarity.\n"
print "                   Est. J   Act. J"

true_pair = 0
false_pair = 0
cadidate_pair = 0
num_pair = (numDocs -1) * numDocs /2
# For each of the document pairs...
for i in range(0, numDocs):
    for j in range(i + 1, numDocs):
        # Retrieve the estimated similarity value for this pair.
        estJ = estJSim[getTriangleIndex(i, j)]

        # If the similarity is above the threshold...
        if estJ >= threshold:
            cadidate_pair += 1
            # Calculate the actual Jaccard similarity for validation.
            s1 = docsAsShingleSets[docNames[i]]
            s2 = docsAsShingleSets[docNames[j]]
            if(len(s1.union(s2)) == 0):
                J = 0
            else:
                J = (len(s1.intersection(s2)) / len(s1.union(s2)))

            if(J >= threshold):
                true_pair += 1
            else:
                false_pair += 1

            # Print out the match and similarity values with pretty spacing.
            #print "  %5s --> %5s   %.2f     %.2f" % (docNames[i], docNames[j], estJ, J)

            # Check whether this is a true positive or false positive.
            # We don't need to worry about counting the same true positive twice
            # because we implemented the for-loops to only compare each pair once.

file_result = open("./result/real_result", "a")
file_result.writelines("\nTotal number of document: " + str(numDocs))
file_result.writelines("\nNumber of band: " + str(numBands) + ". Number of rows: " + str(numRows))
file_result.writelines("\nThresshold t = (1/b)^(1/n) = " + str(pow((1/numBands), (1/numRows))))
file_result.writelines("\nSo choose thresshold t = " + str(threshold))
file_result.writelines("\nTotal cadidate pair found is: " + str(cadidate_pair))

file_result.writelines("\nCompute number of true pair is: " + str(true_pair))
file_result.writelines("\nT = " + str(true_pair / cadidate_pair))

file_result.writelines("\nCompute number of false pair is: " + str(false_pair))
file_result.writelines("\nE = " + str(false_pair / cadidate_pair))