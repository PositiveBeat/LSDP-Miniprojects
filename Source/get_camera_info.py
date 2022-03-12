import exifread
from xml.dom.minidom import parseString

def get_gimbal_orientation(filename):
    yaw = None
    pitch = None
    roll = None

    # Open the file and extract exif information.
    f = open(filename, 'rb')
    # The debug = True option is needed to search for xmp information
    # in the .jgp file. 
    tags = exifread.process_file(f, debug=True)

    if "Image ApplicationNotes" in tags.keys():
        # Extract the xmp values and put them in a dictionary.
        dom = parseString(tags["Image ApplicationNotes"].printable)
        temp = dom.getElementsByTagName("rdf:Description")[0].attributes.items()
        attrs = dict(temp)

        roll = float(attrs['drone-dji:GimbalRollDegree'])
        pitch = float(attrs['drone-dji:GimbalPitchDegree'])
        yaw = float(attrs['drone-dji:GimbalYawDegree'])
    else:
        raise Exception("Could not find gimbal orientation information")

    return (roll, pitch, yaw)


def get_camera_info(filename):
    f = open(filename, 'rb')
    tags = exifread.process_file(f, debug=True)
    for tag in tags.keys():
        if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename'):
            print ("%s: %s" % (tag, tags[tag]))
    


path = "../2019-03-19 Images for third miniproject/"
filename = path + "EB-02-660_0595_0007.JPG"
get_camera_info(filename)
# print(get_gimbal_orientation(filename))