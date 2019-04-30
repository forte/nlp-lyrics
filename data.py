import csv
from langdetect import detect
import random

'''
This file is for manipulating the CSV file of 360k+ songs until there is an individual CSV file
for each genre in this experiment. The functions are meant to be run in chronological order to
do the complete transformation from the original CSV file.
'''


def only_valid_genres():
    '''
    Function that removes all songs from the CSV file that do not have one of the genres below.

    Songs before this function: 362,238
    Songs after this function: 214,786
    '''

    valid_genres = set(['Pop', 'Rock', 'Hip-Hop'])

    with open('lyrics.csv', 'rb') as inp, open('lyrics_v2.csv', 'wb') as out:
        writer = csv.writer(out)
        for row in csv.reader(inp):
            genre = row[4]
            if genre in valid_genres:
                writer.writerow(row)

def only_valid_language_and_len():
    '''
    Function that removes all songs from the CSV file that are not English or have very few lyrics.

    Songs before this function: 214,786
    Songs after this function: 152,963

    *** TAKES A LONG TIME BECAUSE OF LANGUAGE DETECTION ***
    '''

    with open('lyrics_v2.csv', 'rb') as inp, open('lyrics_v3.csv', 'wb') as out:
        writer = csv.writer(out)
        for num, row in enumerate(csv.reader(inp)):
            if num % 10000 == 0:  # to make sure this baby is still chugging along
                print(num)

            try:
                lyrics = row[5].encode('ascii', 'ignore')
                if detect(lyrics) == 'en' and len(lyrics) > 280:
                    writer.writerow(row)
            except:
                continue

def seperate_songs():
    '''
    Function that seperates the different genres into seperate files.

    Number of songs in lyrics_hiphop_v1.csv: 22,069
    Number of songs in lyrics_rock_v1.csv: 97,176
    Number of songs in lyrics_pop_v1.csv: 33,718
    '''

    with open('lyrics_v3.csv', 'rb') as inp, open('lyrics_hiphop_v1.csv', 'wb') as out_hip, \
    open('lyrics_pop_v1.csv', 'wb') as out_pop, open('lyrics_rock_v1.csv', 'wb') as out_rock:

        write_hip = csv.writer(out_hip)
        write_pop = csv.writer(out_pop)
        write_rock = csv.writer(out_rock)

        for row in csv.reader(inp):
            genre = row[4]
        
            if genre == 'Pop':
                write_pop.writerow(row)
            elif genre == 'Rock':
                write_rock.writerow(row)
            elif genre == 'Hip-Hop':
                write_hip.writerow(row)
             
def equalize_song_numbers():
    '''
    Function that randomly selects 21k songs from each genre and puts them in their own CSV file.
    '''

    fillin = ['hiphop', 'rock', 'pop']

    for i in range(3):
        curr_set = set()
        with open('lyrics_' + fillin[i] + '_v1.csv', 'rb') as inp, \
        open('lyrics_' + fillin[i] + '_final.csv', 'wb') as out:
            reader = csv.reader(inp)
            write = csv.writer(out)
            l = list(reader)
            while len(curr_set) < 21000:
                row = random.choice(l)
                n = row[0]
                if n not in curr_set:
                    write.writerow(row)
                curr_set.add(n)
    
def count_songs(suffix):
    '''
    Helper function to see how many songs are in a CSV file.
    '''

    file_name = 'lyrics' + suffix + '.csv'
    with open(file_name, 'rb') as inp:
        num_songs = len(list(csv.reader(inp)))
        print('Number of songs in ' + file_name + ': ' + str(num_songs))
