
heading_list = []
features_list = []

f = open("data.csv", "r")

i = 0
for line in f.readlines():
    print line
    temp = line.split(",")
    print temp
    print "event-id...." + temp[0]
    #print "visible...." + temp[1]
    print "timestamp...." + temp[2]
    print "location-long....." + temp[3]
    print "location-lat....." + temp[4]
    print "eobs:temperature....." + temp[5]
    print "heading....." + temp[6]
    if i == 0 or temp[0] == '':
        i = 1
        continue
    heading = float(temp[6])
    print heading
    if heading < 90 and heading >= 0:
        heading_list.append("north")
    if heading < 180 and heading >= 90:
        heading_list.append("east")
    if heading < 270 and heading >= 180:
        heading_list.append("south")
    if heading < 360 and heading >= 270:
        heading_list.append("west")
    print heading_list

    print "height-above-ellipsoid...." + temp[7]
    #print "tag-local-identifier....." + temp[8]
    #print "individual-local-identifier....." + temp[9]


#x = raw_input()

print heading_list

f.close()
