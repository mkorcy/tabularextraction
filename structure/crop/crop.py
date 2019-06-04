import logging
import cv2
import os
import glob
import shutil
import numpy as np
import ntpath
import pytesseract
import re
from alyn import deskew
from fpdf import FPDF
from PIL import Image
from pdf2image import convert_from_path

# create global logger with 'crop_errors'
logger = logging.getLogger('crop_errors')
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.FileHandler('crop_errors.log')
fh.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)


def convert_pdf_to_images_and_crop(pdf_path, output_path):
    pdf_name = ntpath.basename(pdf_path) 
    pages = convert_from_path(pdf_path, dpi=300)
   
    for page in pages:    
        image_name = "%s-page-%d.jpg" % (pdf_name, pages.index(page))
        jpeg_path = os.path.join(output_path, image_name)
        deskew_path = os.path.join('deskewed_images', image_name)
        page.save(jpeg_path, "PNG")
        
        # Trying to detect rotated images before doing deskew
        erode_image(jpeg_path)
        # Get information about orientation and script detection
        osd_data = pytesseract.image_to_osd(Image.open(os.path.join('erode_images',image_name)))
        
        # if the rotation of the document looks completely off, rotate as deskewing it won't help if its
        # off by a very large # of degrees 90,180, etc.
        rotation = re.search('(?<=Rotate: )\d+', osd_data).group(0)
        confidence = re.search('(?<=Orientation confidence: )\d+', osd_data).group(0)
        rotation = int(rotation)
        confidence = int(confidence)
        
        if rotation > 0 and confidence > 5:        
            image_to_rotate  = Image.open(jpeg_path)
            image_to_rotate = image_to_rotate.rotate(360-rotation)
            image_to_rotate.save(jpeg_path,"PNG")
        
        # https://github.com/mawanda-jun/Alyn
        # Python3 compatible fork of alyn
        d = deskew.Deskew(
            input_file=jpeg_path,            
            output_file=deskew_path,
            r_angle=0)
        d.run()

        
        # Crop the images down via the horizontal line
        first_crop('deskewed_images', "%s-page-%d.jpg" % (pdf_name, pages.index(page)), output_dir="first_crop_images", pdf_name=pdf_name, page_number=pages.index(page))
        second_crop('first_crop_images', "%s-page-%d.jpg" % (pdf_name, pages.index(page)), output_dir="second_crop_images", pdf_name=pdf_name, page_number=pages.index(page))
    
    # once we have stacks of images that have been cropped
    # restack them back into a single PDF for easier processing

    for page in pages:
        # Covert the cropped images back to pdf form
        second_crop_page = os.path.join("second_crop_images", "%s-page-%d.jpg" % (pdf_name, pages.index(page)))        

def erode_image(jpeg_path):
    image_name = ntpath.basename(jpeg_path) 
    image = cv2.imread(jpeg_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray,(3,3))
    _,thresh = cv2.threshold(blur,240,255,cv2.THRESH_BINARY)
    #cv2.imshow("thresh",thresh)
    thresh = cv2.bitwise_not(thresh)
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
    erode = cv2.erode(thresh,element,3)
    
    cv2.imwrite(os.path.join('erode_images',image_name), erode)
#
#   Create a PDF from a stack of images
#

def make_pdf(pdfFileName, listPages, dir = ''):
    if (dir):
        dir += "/"
     
    cover = Image.open(listPages[0])
    width, height = cover.size

    pdf = FPDF(unit = "pt", format = [width, height+150])

    for page in listPages:
        pdf.add_page()
        pdf.image(page, 0, 150)

    pdf.output(dir + pdfFileName, "F")

#
#   Crop the specified image to the first horizontal line.
#

def first_crop(input_dir, image_name, output_dir, pdf_name, page_number):

    img = cv2.imread(os.path.join(input_dir, image_name), 0)
    height, width = img.shape[:2]

    edges = cv2.Canny(img, 100, 200)

    lines = cv2.HoughLines(edges, 1, np.pi/90, 400)
    if lines is None:
        logger.error("First Crop failed PDF: %s Page: %s" %(pdf_name, page_number,))
        cv2.imwrite(str(os.path.join(output_dir, image_name)), img)
    else:        
        for rho,theta in lines[0]:

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255),2)
            crop_img = img[y1+6:height, 0:width]
        
            cv2.imwrite(str(os.path.join(output_dir, image_name)), crop_img)

        # To Show Matplot graph with drawn line, uncomment

        # plt.subplot(121),plt.imshow(img, cmap = 'gray')
        # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        # plt.plot(122),plt.imshow(edges,cmap = 'gray')
        # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        # plt.show()


#
#   Crop the specified image to the second horizontal line
#   (After first Crop has been called)
#

def second_crop(input_dir, image_name, output_dir, pdf_name, page_number):

    img = cv2.imread(os.path.join(input_dir, image_name), 0)
    height, width = img.shape[:2]

    edges = cv2.Canny(img, 100, 200)

    lines = cv2.HoughLines(edges, 1, np.pi / 90, 100)
    if lines is None:
        logger.error("Second Crop failed PDF: %s Page: %s" %(pdf_name, page_number,))
        cv2.imwrite(str(os.path.join(output_dir, image_name)), img)
    else:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


            crop_img = img[y1 + 6:height, 0:width]
            path = str(os.path.join(output_dir, image_name))

            cv2.imwrite(path, crop_img)

# find files regardless of case to deal with *.PDF file endings when I would
# normally expect *.pdf
def insensitive_glob(pattern):
        def either(c):
            return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
        return glob.glob(''.join(map(either, pattern)))

def prep_directories():
    dirname = os.path.dirname(__file__)
    output_dirs = ['input_pdfs', 'output_images','deskewed_images','erode_images','output_pdfs','first_crop_images','second_crop_images']
    for dir in output_dirs:
        os.makedirs( os.path.join(dirname, dir), exist_ok=True);

def breakdown_scratch_files():
    dirname = os.path.dirname(__file__)
    dirs_to_clean = ['output_images','deskewed_images','erode_images','first_crop_images','second_crop_images']
    for dir in dirs_to_clean:
        shutil.rmtree(os.path.join(dirname, dir), ignore_errors=True, onerror=None)
    
def process_pdfs():
    input_pdfs = insensitive_glob('./input_pdfs/*.pdf')
    for input_pdf in input_pdfs:
        # Input Directory, Name of Pdf, Page to Convert, Place to Save Image
        convert_pdf_to_images_and_crop(input_pdf, output_path='output_images/')                  
        pdf_name = ntpath.basename(input_pdf)
        input_images = insensitive_glob('./second_crop_images/*.jpg')
        input_images.sort()
        print(input_images)
        make_pdf(pdf_name, input_images, dir = 'output_pdfs/')
#
#   Example conversion of particular pdf page to cropped image and
#   then "cropped pdf" equivalent.
#

if __name__ == '__main__':
    # get the working directory so we can create folders
    prep_directories()

    process_pdfs()
    
    breakdown_scratch_files()