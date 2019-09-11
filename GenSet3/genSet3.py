#!/usr/bin/env python
# coding: utf8
# version 3.0
# -------------------------------------------
# ---------------- GenSet V3 ----------------
# -------------------------------------------

from PIL import Image, ImageFont, ImageDraw
from collections import defaultdict
import random
import os
import re
import errno
from sys import argv, stdout
import string
import numpy as np

import math
import matplotlib.pyplot as plt

try:
    import progressbar
except ImportError:
    print("pip install progressbar2")
    exit()

# Anzahl Cluster für Farbclustering. 
# Bei höherer Anzahl an Clustern sind mehr Farben dabei,
# wobei von diesen die dunkelsten R/G/B-Werte genommen werden,
# aus denen dann die Schriftfarbe gebildet wird
# Clustering wird einmal je Hintergrund durchgeführt
KMeans_nClusters = 5 

charDict = defaultdict(lambda: defaultdict(lambda: 0))
backgrounds = []
fonts = []


def initFonts_manual(fontPath='./fonts/'): # OLD
    def add(name, file):
        fonts.append((name, fontPath + file))
    add("Times New Roman", "times-new-roman-591b0445f2883.ttf")
    add("Roboto-Regular", "GoogleFonts/Roboto/Roboto-Regular.ttf")
    add("Roboto-Bold", "GoogleFonts/Roboto/Roboto-Bold.ttf")
    add("Lora-Regular", "GoogleFonts/Lora/Lora-Regular.ttf")
    add("Bree-Serif-Regular", "GoogleFonts/Bree_Serif/BreeSerif-Regular.ttf")
    add("LibreBaskerville-Regular", "GoogleFonts/Libre_Baskerville/LibreBaskerville-Regular.ttf")
    add("im-fell-french-canon-Cit2", "im-fell-french-canon/FeFCit2.ttf")
    add("im-fell-french-canon-Crm2", "im-fell-french-canon/FeFCrm2.ttf")
    add("im-fell-french-canon-Csc2", "im-fell-french-canon/FeFCsc2.ttf")
    add("junicode", "junicode/Junicode.ttf")
    add("junicode", "junicode/Junicode-Italic.ttf")
    add("dominican", "dominican/DOMINICA.ttf")
    add("OldNewsPaperTypes", "manfred-klein_old-newspaper-types/OldNewspaperTypes.ttf")

def initFonts(fontPath='./fonts/'):
    for root, dirs, files in os.walk(fontPath, topdown=True):
        for filename in files:
            split = os.path.splitext(os.path.split(filename)[1])

            if(split[1] == ".ttf"): # use only .ttf files
                path = os.path.join(root, filename).replace("\\", "/") # just some naming magic
                fonts.append((split[0], path))

def loadBackgroundImages():
    for file in os.listdir("background/"):
        if file.endswith(".png"):
            backgrounds.append(Image.open("background/" + file))



def loadCharImages():
    # bisher Hardcoded
    # TODO: bessere Lösung entwickeln

    # charDict[Buchstabe][Typ]

    charDict[1]['A'] = Image.open("char/25n_map1_Ackendorf/A.png")
    charDict[1]['B'] = Image.open("char/25n_map1_Berge/B.tif")
    charDict[1]['C'] = Image.open("char/29n2_map4_Cleverns/scaled_26n/C.png")
    charDict[1]['D'] = Image.open("char/29n2_map2_Dykhausen/scaled_25n/D.png")
    charDict[1]['E'] = Image.open("char/30n_map3_Eckwarden/scaled_25n/E.png")
    charDict[1]['G'] = Image.open("char/25n_2333_Gross-Bruetz_1879-1881_s1_Gottesgabe/G.png")
    charDict[1]['I'] = Image.open("char/25n_map1_Ipse/I.png")
    charDict[1]['J'] = Image.open("char/23n_map1_J+aevenitz/J.png")
    charDict[1]['K'] = Image.open("char/29n_1282_Leer_1897_1898_s1_Kritzum/scaled_25n/K.png")
    charDict[1]['L'] = Image.open("char/30n2_map156_Lauenf+oerde/scaled_25n/L.png")
    charDict[1]['M'] = Image.open("char/29n_1282_Leer_1897_1898_s1_Midlum/scaled_25n/M.png")

    charDict[1]['a'] = Image.open("char/25n_2333_Gross-Bruetz_1879-1881_s1_Gottesgabe/a.png")
    charDict[1]['b'] = Image.open("char/25n_2333_Gross-Bruetz_1879-1881_s1_Gottesgabe/b.png")
    charDict[1]['c'] = Image.open("char/25n_map1_Ackendorf/c.png")
    charDict[1]['d'] = Image.open("char/25n_map1_Ackendorf/d.png")
    charDict[1]['e'] = Image.open("char/25n_map1_Ackendorf/e.png")
    charDict[1]['f'] = Image.open("char/25n_map1_Ackendorf/f.png")
    charDict[1]['g'] = Image.open("char/25n_map1_Berge/g.tif")
    charDict[1]['h'] = Image.open("char/30n_6117_Darmstadt_bearbeitet1886_s1_Griesheim/scaled_25n/h.png")
    charDict[1]['i'] = Image.open("char/30n_6117_Darmstadt_bearbeitet1886_s1_Griesheim/scaled_25n/i.png")
    # charDict[1]['j'] = []
    charDict[1]['k'] = Image.open("char/25n_map1_Ackendorf/k.png")
    charDict[1]['l'] = Image.open("char/23n_map1_Algensted/l.png")
    charDict[1]['m'] = Image.open("char/30n_6117_Darmstadt_bearbeitet1886_s1_Griesheim/scaled_25n/m.png")
    charDict[1]['n'] = Image.open("char/25n_map1_Ackendorf/n.png")
    charDict[1]['o'] = Image.open("char/25n_map1_Ackendorf/o.png")
    # charDict[1]['p'] = []
    charDict[1]['z'] = Image.open("char/25n_map1_Ipse/p.png")
    charDict[1]['r'] = Image.open("char/25n_map1_Ackendorf/r.png")
    charDict[1]['s'] = Image.open("char/30n_6117_Darmstadt_bearbeitet1886_s1_Griesheim/scaled_25n/s.png")
    charDict[1]['t'] = Image.open("char/25n_2333_Gross-Bruetz_1879-1881_s1_Gottesgabe/t.png")
    charDict[1]['u'] = Image.open("char/30n2_map156_Lauenf+oerde/scaled_25n/u.png")
    charDict[1]['v'] = Image.open("char/29n2_map4_Cleverns/scaled_26n/v.png")
    charDict[1]['w'] = Image.open("char/30n_map3_Eckwarden/scaled_25n/w.png")
    # charDict[1]['x'] = []
    charDict[1]['y'] = Image.open("char/29n2_map2_Dykhausen/scaled_25n/y.png")
    charDict[1]['z'] = Image.open("char/23n_map1_J+aevenitz/z.png")


def HDisplacement(A, B):
    # Berechnen der Verschiebung für ein Bild in Relation zu "Ankerbild"
    # Problem: Pillow will linke obere Ecke --> verschiedene Höhen der Bilddateien sind problematisch
    # --> Berechnung einer Verschiebung, sodass untere Kante übereinstimmt
    displ = A.height - B.height
    return displ


def calcBackgroundSize(str2draw, set, offsetX=5, offsetY=5, spacingX=15):
    # Zur Berechnung der notwendigen Größe des Hintergrundes
    # spacingX - Abstand zwischen den einzufügenden Buchstaben
    # offsetX - Abstand erster/letzter Buchstabe zum linken/rechten Rand
    # offsetY - Abstand höchster Buchstabe zum oberen/unteren Rand
    height_min = 0  # Mindesthöhe
    width_min = 0  # Mindestbreite

    for c in str2draw:
        char_img = charDict[set][c]
        if (char_img.height > height_min):
            # neue Mindesthöhe anhand höchstem height-Wert aus den einzufügenden Char-Images
            height_min = char_img.height

        # Mindestbreite anpassen
        # TODO: letzten Buchstaben davon ausnehmen
        width_min += char_img.width + spacingX

    # Offsets addieren
    width = width_min + offsetX * 2
    height = height_min + offsetY * 2

    return (width, height)


def drawString(background, str2draw, set, offsetX=5, offsetY=5, spacingX=15):
    # Zerlegung des Strings in Buchstaben -> Raussuchen der Buchstaben aus dem richtigen Set, Berechnen der Abstände und Zeichnen

    # spacingX - Abstand zwischen den einzufügenden Buchstaben
    # offsetX - Abstand erster/letzter Buchstabe zum linken/rechten Rand
    # offsetY - Abstand höchster Buchstabe zum oberen/unteren Rand

    # Startposition
    x_start = offsetX
    y_start = offsetY

    # Background-Image handlen
    img = Image.new('RGBA', (background.width, background.height), (255, 0, 0, 0))
    img.paste(background, (0, 0))

    highestHeight = 0
    lowestHeight = 999

    # TODO: fixen!
    for c in str2draw:
        char_img = charDict[set][c]
        if (char_img.height > highestHeight):
            # neuer Höchstwert
            highestHeight = char_img.height

        if (char_img.height < lowestHeight):
            lowestHeight = char_img.height

    # für HDisplacement --> erster Buchstabe als "Ankerbild"
    char_img_start = charDict[set][str2draw[0]]
    y_start = (background.height - charDict[set][str2draw[0]].height) // 2

    x = x_start
    y = y_start

    for c in str2draw:
        char_img = charDict[set][c]
        img.paste(char_img, (x, y + HDisplacement(char_img_start, char_img)), char_img)

        # neue Position
        x += char_img.width + spacingX

    return img


def getRandomBackgroundArea(backgroundImage, targetWidth, targetHeight):
    # schneidet zufälligen Abschnitt aus gegebenem Hintergrund aus
    # TODO: überprüfen, ob gegebener Background überhaupt groß genug

    randomOffsetX_max = backgroundImage.width - targetWidth
    randomOffsetY_max = backgroundImage.height - targetHeight

    if(randomOffsetX_max < 0 or randomOffsetY_max < 0):
        # Hintergrund zu klein!
        return None

    randomOffsetX = random.randint(0, randomOffsetX_max)
    randomOffsetY = random.randint(0, randomOffsetY_max)

    rndBackgroundArea = backgroundImage.crop((randomOffsetX, randomOffsetY, randomOffsetX + targetWidth, randomOffsetY + targetHeight))
    return rndBackgroundArea


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def saveLabeledImage(image, label, set, backgroundName, basepath, printOutput=True, number=''):
    path = basepath
    make_sure_path_exists(path)
    if(number != ''):
        number = '_' + number
    filename = os.path.join(path, label + '.jpg')
    rgbimage = image.convert("RGB")
    rgbimage.save(filename, "JPEG")
    if(printOutput):
        print("Saved image to " + filename)


def randomString(charList, minLength, maxLength, uppercaseBehaviour='random'):
    # erzeugt einen zufälligen String aus den verfügbaren Chars der jeweiligen Liste
    # upperCBehaviour = ['random', 'onlyfirstMAX','onlyfirstMIN', 'allUpper']
    # andere noch nicht implementiert
    upperCBehaviour = ['random', 'onlyfirstMAX', 'onlyfirstMIN']
    if uppercaseBehaviour not in upperCBehaviour:
        raise ValueError("Invalid uppercaseBehaviour. Expected one of: %s" % upperCBehaviour)

    length = minLength + random.randint(0, maxLength - minLength)
    rndString = ""

    # TODO: ACHTUNG! MOMENTAN PROBLEM, WENN NICHT MIND. 1 GROß- UND KLEINBUCHSTABE!

    def isLowerCase(c):
        if(c == c.lower()):
            return True
        return False

    def isUpperCase(c):
        if(c == c.upper()):
            return True
        return False

    def getSampleLowerCase():
        sample = 'X'
        while(sample != sample.lower()):
            sample = random.sample(charList, 1)[0]
        return sample

    def getSampleUpperCase():
        sample = 'x'
        while(sample != sample.upper()):
            sample = random.sample(charList, 1)[0]
        return sample

    for i in range(length):
        if(uppercaseBehaviour == 'random'):
            rndString += random.sample(charList, 1)[0]

        elif(uppercaseBehaviour == 'onlyfirstMAX'):
            # maximal 1. Buchstabe groß
            if(i == 0):
                rndString += random.sample(charList, 1)[0]
            else:
                rndString += getSampleLowerCase()
        elif(uppercaseBehaviour == 'onlyfirstMIN'):
            # erster Buchstabe ist garantiert groß, keine weiteren Großbuchstaben
            if(i == 0):
                rndString += getSampleUpperCase()
            else:
                rndString += getSampleLowerCase()
    return rndString


def load_all_words_from_file(input_file):
    with open(input_file, "r") as f:
        lines = f.readlines()
    
    return lines

class WordGenerator():
    def __init__(self):
        self.dictionary_iterations = 0
        self.current_word_counter = 0

    def getRandomWord(self, dictionary, pattern):
        while True:
            if((self.current_word_counter) == len(dictionary)):
                self.dictionary_iterations += 1
                self.current_word_counter = 0

            word = dictionary[self.current_word_counter]
            self.current_word_counter += 1
            result = pattern.match(word)
            
            if result is not None:
                break
        word = result.group(0) + self.randomString(self.dictionary_iterations)
        return word
    
    def randomString(self, stringLength=10):
        """Generate a random string of fixed length """
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(stringLength))

def loadDictFile(file):

    print("Using dictionary file " + file)
    words = []
    f = open(file, "r")

    for line in f:
        words.append(line.replace("\n", ""))

    print(len(words))

    return words


def spacingTest(word, printIfTooSmall=False):
    # Test für Spacing zwischen Wörtern

    loadCharImages()
    loadBackgroundImages()
    backnr = 0
    for back in backgrounds:
        backnr += 1
        for spacing in [1, 2, 5, 10, 15, 20, 25]:
            width, height = calcBackgroundSize(word, 1, spacingX=spacing, offsetY=0)
            backgroundArea = getRandomBackgroundArea(back, width, height)
            if(backgroundArea is not None):
                img = drawString(backgroundArea, word, 1, spacingX=spacing, offsetY=0)
                saveLabeledImage(img, word + '_' + str(spacing), 1, "back_" + str(backnr), './spacingTest/', printOutput=False)
            else:
                if printIfTooSmall:
                    print("Background " + str(backnr) + " too small for word " + word + " with given offsetX " + str(spacing) + "!")

def getWordPattern():
    return re.compile("[a-zA-zÄäÜüÖöß ]+")

def generateRandomSamples(nSamples_randomFonts, dataset_name, printIfTooSmall=False, wordList=None):
    # savePath = '../labeled/generated/'
    if(dataset_name == None):
        print("Error: Don't forget to name the dataset!")
        exit()

    savePath = os.path.abspath(dataset_name)

    savePathNoBackground=os.path.join(savePath, "NoBackground")
    savePathWithBackground=os.path.join(savePath, "WithBackground")

    loadBackgroundImages()


    # Schritt 1: charDict befüllen, Hintergründe laden
    # loadCharImages()
    # nSamples_per_background_cutChars = nSamples_cutChars // len(backgrounds)
    # nSamples_cutChars = nSamples_per_background_cutChars * len(backgrounds)
    # print("Generating " + str(nSamples_per_background_cutChars) + " samples per background with " + str(len(backgrounds)) + " backgrounds with cut chars\n")
    # stdout.flush()
    # bar = progressbar.ProgressBar(max_value=len(backgrounds) * nSamples_per_background_cutChars)
    # bar.update(0)
    # backnr = 0
# 
    # skipped = 0  # total number of skipped words (because too small...)
# 
    # Variante 1: Buchstabensalat
    # for back in backgrounds:
    #     backnr += 1
# 
    #     for i in range(nSamples_per_background_cutChars):
    #         # Schritt 2: benötigte Größe berechnen, zufälligen Bereich dieser Größe aus einem der Hintergrund-Files auswählen
    #         # TODO: nur Hintergründe betrachten, die groß genug sind
    #         set = 1
# 
    #         # TODO: Kompatibilität für Dictionary-Variante
    #         # if(wordList is None):
    #         label = randomString(charDict[set].keys(), 2, 10, uppercaseBehaviour='onlyfirstMAX')
    #         # else:
    #         # label = randomWord(wordList, 2, 10, uppercaseBehaviour='onlyfirstMAX')
# 
    #         # Abstand zwischen Buchstaben
    #         spacingX = 10 + random.randint(-8, 2)
# 
    #         offsetY = 0
    #         offsetX = 30 + random.randint(-20, 30)
    #         width, height = calcBackgroundSize(label, 1, spacingX=spacingX, offsetY=offsetY, offsetX=offsetX)
    #         backgroundArea = getRandomBackgroundArea(back, width, height)
# 
    #         # Schritt 3: Beschriftung draufpacken!
    #         if(backgroundArea is not None):
    #             img = drawString(backgroundArea, label, 1, spacingX=spacingX, offsetY=offsetY, offsetX=offsetX)
    #             saveLabeledImage(img, label, 1, "cutChars_back_" + str(backnr), savePath, printOutput=False)
    #         else:
    #             skipped += 1
    #             if printIfTooSmall:
    #                 print("\nBackground " + str(backnr) + " too small for word " + label + " with given offsetX " + str(spacingX) + "!")
    #         # TODO: mehrere gleiche Label beim Trainieren ermöglichen -> number-Suffix darf nicht mit zum Label gehören!
    #     bar.update(backnr * nSamples_per_background_cutChars)

    # --
    # Variante 2: Schriftartensalat
    # --
    initFonts()
    nSamples_per_background_randomFonts = nSamples_randomFonts // len(backgrounds)
    nSamples_randomFonts = nSamples_per_background_randomFonts * len(backgrounds)
    print("\n\nGenerating " + str(nSamples_per_background_randomFonts) + " samples per background with " + str(len(backgrounds)) + " backgrounds with " + str(len(fonts)) + " random fonts\n")
    stdout.flush()
    bar2 = progressbar.ProgressBar(max_value=len(backgrounds) * nSamples_per_background_randomFonts)
    bar2.update(0)
    backnr = 0
    minSize = 20
    maxSize = 80
    minScale = 1.4
    maxScale = 3

    dictionary = load_all_words_from_file("cities.txt")
    pattern = getWordPattern()
    word_generator = WordGenerator()

    for back in backgrounds:
        backnr += 1 

        cvFontColor = getFontColor(back) # Berechnen der zweitdominantesten Farbe im Background -> Schriftfarbe

        for i in range(nSamples_per_background_randomFonts):
            label = word_generator.getRandomWord(dictionary, pattern)
            fontSize = random.randint(minSize, maxSize)
            i = random.randint(0, len(fonts) - 1)
            #print(fonts[i][1])
            try:
                font = ImageFont.truetype(fonts[i][1], fontSize)
            except OSError:
                print("Error  @ " + fonts[i][1])
                break
            size = font.getsize(label)
            scale_factor = random.uniform(minScale, maxScale)
            font_img = Image.new('RGBA', (size[0], size[1]), (255, 255, 255, 0))
            # ImageDraw.Draw(font_img).text((0, 0), label, font=font, fill=randomFontColor())
            ImageDraw.Draw(font_img).text((0, 0), label, font=font, fill=cvFontColor)
            newSize = (int(size[0] * scale_factor), int(size[1] * scale_factor))
            font_img = font_img.resize(newSize)

            backgroundArea = getRandomBackgroundArea(back, int((size[0] + fontSize // 3) * scale_factor), int((size[1] + fontSize // 3) * scale_factor))
            if(backgroundArea is not None):
                img = Image.new('RGBA', (backgroundArea.width, backgroundArea.height), (255, 255, 255, 0))
                img.paste(font_img, (0, 0), font_img)
                # dirname should not contain spaces because of parsing-the-annotation-files related things -> replace them!
                fontDirName = fonts[i][0].replace(' ', '_')

                saveLabeledImage(img, label, fontDirName, "back_" + str(backnr), savePathNoBackground, printOutput=False)

                for i in range(1,font_img.width-1):
                    for j in range(1,font_img.height-1):
                        (r, g, b, a) = font_img.getpixel((i,j))

                        if(a > 0):
                            r += (random.randint(-10, 30));
                            g += (random.randint(-10, 30));
                            b += (random.randint(-10, 30));

                        font_img.putpixel((i,j), (r, g, b, a))

                # Add background and save it again
                img.paste(backgroundArea, (0, 0)) # Paste background
                img.paste(font_img, (0, 0), font_img)
                saveLabeledImage(img, label, fontDirName, "back_" + str(backnr), savePathWithBackground, printOutput=False)

        bar2.update(backnr * nSamples_per_background_randomFonts)

    print("\nDone.")


def fontTest():
    loadBackgroundImages()
    fontSize = 128
    font = ImageFont.truetype('./fonts/times-new-roman-591b0445f2883.ttf', fontSize)
    size = font.getsize("hello")
    backgroundArea = getRandomBackgroundArea(backgrounds[0], size[0] + fontSize // 3, size[1] + fontSize // 3)
    draw = ImageDraw.Draw(backgroundArea)
    draw.text((fontSize // 6, 0), "hello", font=font, fill=(90, 90, 90, 0))
    backgroundArea.show()


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        print(percent, color)
        startX = endX

    # return the bar chart
    return bar


def getFontColor(back):
    return (math.floor(120 * random.uniform(0.9, 1.1)), math.floor(110 * random.uniform(0.9, 1.1)), math.floor(100 * random.uniform(0.9, 1.1)), 255)

if __name__ == '__main__':
    # spacingTest("Dykhausen")
    print("\n \n--------------------------------------------")
    print("---------------- GenSet 3.0 ----------------")
    print("--------------------------------------------\n \n")
    stdout.flush()

    dictFile = "./english-words/words_alpha.txt"

    if(len(argv) >= 3):
        # nSamples_cutChars = int(argv[1])
        nSamples_randomFonts = int(argv[1])

        dataset_name = argv[2]

        useDict = False
        if(len(argv) > 3):
            if(argv[3] == "--usedict"):
                useDict = True
        
        wordList = None
        if(useDict == True):
            wordList = loadDictFile(dictFile)
        # generateRandomSamples(nSamples_cutChars, nSamples_randomFonts, dataset_name, wordList=wordList)
        generateRandomSamples(nSamples_randomFonts, dataset_name, wordList=wordList)
    else:
        print("Usage:\ngenSet.py <nSamples_cutChars> <nSamples_randomFonts> <dataset name> [--usedict]\nUsed for generating synthetic training data.\nCreates an equal amount of samples for each given background, so the total amount of samples will likely be a bit lower than the given nSamples.")
        print("With --usedict option: use dictionary for words")
        exit()
