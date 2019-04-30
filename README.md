# Using Natural Language Processing to Analyze Lyrics and Song Classification

## Goal

<b>*Developing a system that can predict the genre of a song by solely looking at its lyrics.*</b> 
<br>
<br>
Such a system would offer insight into: 
- The homogeneity of songs across a genre 
- Surprising similarities between genres
- The extent to which instrumentals define a song’s genre

Motivation:
- Working with a unique and complex dataset
- Exploring the insight of such a model
- Building a classification system


## Data Normalization

### Songs to include:
- Has genre that is Pop, Rock, or Hip-Hop
- Minimum length lyrics
- Only songs that are English

### Lyrics alterations:
- Remove hings that signify chorus, verse, etc...
  - Typically these are in [] but can also be in (), {}
  - This can be problematic because valuable lyrics can also be present in (), but it is best to play it safe and remove these things.
- Remove ung noises (‘laaa’, ‘oooo’, 'ahhhhh')
- Leveraging lemmatization using the nltk library
- Extending from the nltk library’s stopword list to include slang / mispelled words


## About the Data

- The [360k song corpus](https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics) that was originally leveraged for this experiment was transformed into a 63k song dataset evenly containing Pop, Rock, and Hip-Hop songs that were normalized using the methods mentioned above.
- The code for the transformation from the original dataset to the one used in experiments can be seen in `data.py`.

**Curse words present per song:**
Pop: **0.1%**
Rock: **0.2%**
Hip-Hop: **3.6%**

**Type-Token Ratio:**
Pop: **0.54**
Rock: **0.61**
Hip-Hop: **0.62**

**Number of Tokens:**
Pop: **104**
Rock: **85**
Hip-Hop: **235**

- The inherent differences between Hip-Hop and the other two genres (significantly higher percentage of curse words and more data) are probably a major reason why Hip-Hip is much easier to predict with this data.

## Experiment Trials:
- For each of the 3 classifiers below, 6 models were run with combinations of the following inputs:
  - *n-grams*: unigrams, bigrams, trigrams OR bigrams, trigrams
  - *minimum frequency of n-gram*: 2, 10, or 20 minimum

### Naive Bayes

```Python
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
```
  
**Trial 1:**
n-grams: (1, 3)
min frequency: 2
Features shape: (50393, 1276533)
Hiphop accuracy: 0.898698884758
Rock accuracy: 0.556148435605
Pop accuracy: 0.540508149569
Complete accuracy: 0.667989522978

**Trial 2:** 
n-grams: (2, 3)
min frequency: 2
Features shape: (50393, 1248222)
Hiphop accuracy: 0.828982035928
Rock accuracy: 0.487428842505
Pop accuracy: 0.714353612167
Complete accuracy: 0.676402889118

**Trial 3:** 
n-grams: (1, 3)
min frequency: 10
Features shape: (50393, 90620)
Hiphop accuracy: 0.811904761905
Rock accuracy: 0.638030888031
Pop accuracy: 0.612220916569
Complete accuracy: 0.687276767997

**Trial 4:** 
n-grams: (2, 3)
min frequency: 10
Features shape: (50393, 73508)
Hiphop accuracy: 0.779798946865
Rock accuracy: 0.582519647535
Pop accuracy: 0.662719090478
Complete accuracy: 0.674815461545

**Trial 5:**
n-grams: (1, 3)
min frequency: 20
Features shape: (50393, 41085)
Hiphop accuracy: 0.813692748092
Rock accuracy: 0.662449271903
Pop accuracy: 0.59862494073
Complete accuracy: 0.691404079689

**Trial 6:** 
n-grams: (2, 3)
min frequency: 20
Features shape: (50393, 29422)
Hiphop accuracy: 0.770886964662
Rock accuracy: 0.576368876081
Pop accuracy: 0.620374819798
Complete accuracy: 0.656877529963

### LinearSVC

```Python
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=0)
```

**Trial 1:**
n-grams: (1, 3)
min frequency: 2
Features shape: (50393, 1301348)
Hiphop accuracy: 0.920612485277
Rock accuracy: 0.577216396568
Pop accuracy: 0.619528619529
Complete accuracy: 0.706881498532

**Trial 2:** 
n-grams: (2, 3)
min frequency: 2
Features shape: (50393, 1244192)
Hiphop accuracy: 0.868313744303
Rock accuracy: 0.578600047136
Pop accuracy: 0.640792930499
Complete accuracy: 0.695134534487

**Trial 3:**
n-grams: (1, 3)
min frequency: 10
Features shape: (50393, 90877)
Hiphop accuracy: 0.919697339517
Rock accuracy: 0.520047732697
Pop accuracy: 0.632189239332
Complete accuracy: 0.688387967299

**Trial 4:** 
n-grams: (2, 3)
min frequency: 10
Features shape: (50393, 73518)
Hiphop accuracy: 0.834052757794
Rock accuracy: 0.57951033991
Pop accuracy: 0.602321174799
Complete accuracy: 0.671402492261

**Trial 5:** 
n-grams: (1, 3)
min frequency: 20
Features shape: (50393, 41190)
Hiphop accuracy: 0.911878319652
Rock accuracy: 0.562869997632
Pop accuracy: 0.629664619745
Complete accuracy: 0.700055559965

**Trial 6:** 
n-grams: (2, 3)
min frequency: 20
Features shape: (50393, 29600)
Hiphop accuracy: 0.825060240964
Rock accuracy: 0.556193208991
Pop accuracy: 0.584016873682
Complete accuracy: 0.654178903088

### Logistic Regression

```Python
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='warn', n_jobs=None, penalty='l2', random_state=0, solver='warn', tol=0.0001, verbose=0, warm_start=False)
```

**Trial 1:** 
n-grams: (1, 3)
min frequency: 2
Features shape: (50393, 1284334)
Hiphop accuracy: 0.921885833141
Rock accuracy: 0.516520085572
Pop accuracy: 0.606150061501
Complete accuracy: 0.684657512501

**Trial 2:** 
n-grams: (2, 3)
min frequency: 2
Features shape: (50393, 1250221)
Hiphop accuracy: 0.842143027984
Rock accuracy: 0.572167371885
Pop accuracy: 0.646253602305
Complete accuracy: 0.686244940075

**Trial 3:**
n-grams: (1, 3)
min frequency: 10
Features shape: (50393, 90449)
Hiphop accuracy: 0.915552427868
Rock accuracy: 0.476463560335
Pop accuracy: 0.636232233197
Complete accuracy: 0.677672831177

**Trial 4:** 
n-grams: (2, 3)
min frequency: 10
Features shape: (50393, 73180)
Hiphop accuracy: 0.837003350886
Rock accuracy: 0.603908484271
Pop accuracy: 0.618461538462
Complete accuracy: 0.686086197317

**Trial 5:** 
n-grams: (1, 3)
min frequency: 20
Features shape: (50393, 41119)
Hiphop accuracy: 0.903147109051
Rock accuracy: 0.511661113755
Pop accuracy: 0.655653792462
Complete accuracy: 0.688149853163

**Trial 6:** 
n-grams: (2, 3)
min frequency: 20
Features shape: (50393, 29522)
Hiphop accuracy: 0.819541616405
Rock accuracy: 0.582461977186
Pop accuracy: 0.600329722091
Complete accuracy: 0.666481466783
