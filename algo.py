import time


def schedule_algo():

    total = 20
    initial1 = 5
    initial2 = 0
    initial3 = 0
    initial4 = 0

    mins = 2

    file1 = open('cam1.txt', 'r')
    file2 = open('cam2.txt', 'r')
    file3 = open('cam3.txt', 'r')
    file4 = open('cam4.txt', 'r')

    out1 = open('out1.txt', 'w')
    out2 = open('out2.txt', 'w')
    out3 = open('out3.txt', 'w')
    out4 = open('out4.txt', 'w')

    start = time.time()

    prev = 0
    total_iters = 0

    # while(len(file1.read().splitlines()) == 0):
    #     print('sleep')
    #     time.sleep(1)
    # file1.seek(0, 0)

    while(1):

        times = int((time.time()-start))

        if (times != prev or total_iters == 0):

            if(initial1 == 0):
                if(initial4 == 0):
                    out1.write("r")
                    out1.write('\n')
                    out1.flush()
                else:
                    out1.write("r" + ',' + str(initial4 - times - 1))
                    out1.write('\n')
                    out1.flush()
            else:

                out1.write("g" + ',' + str(initial1 - times - 1))
                out1.write('\n')
                out1.flush()

            if(initial2 == 0):
                if(initial1 == 0):
                    out2.write("r")
                    out2.write('\n')
                    out2.flush()
                else:
                    out2.write("r" + ',' + str(initial1 - times - 1))
                    out2.write('\n')
                    out2.flush()
            else:
                out2.write("g" + ',' + str(initial2 - times - 1))
                out2.write('\n')
                out2.flush()

            if(initial3 == 0):
                if(initial2 == 0):
                    out3.write("r")
                    out3.write('\n')
                    out3.flush()
                else:
                    out3.write("r" + ',' + str(initial2 - times - 1))
                    out3.write('\n')
                    out3.flush()
            else:
                out3.write("g" + ',' + str(initial3 - times - 1))
                out3.write('\n')
                out3.flush()

            if(initial4 == 0):
                if(initial3 == 0):
                    out4.write("r")
                    out4.write('\n')
                    out4.flush()
                else:
                    out4.write("r" + ',' + str(initial3 - times - 1))
                    out4.write('\n')
                    out4.flush()
            else:
                out4.write("g" + ',' + str(initial4 - times - 1))
                out4.write('\n')
                out4.flush()

        total_iters += 1

        prev = times

        if ((times + 1 == initial1 or times+1 == initial2 or times + 1 == initial3 or times+1 == initial4)):

            # while(len(file1.read().splitlines()) == 0):
            #     print('sleep')
            #     time.sleep(1)

            file1.seek(0, 0)
            file2.seek(0, 0)
            file3.seek(0, 0)
            file4.seek(0, 0)

            line1 = int(file1.read().splitlines()[-1])
            line2 = int(file2.read().splitlines()[-1])
            line3 = int(file3.read().splitlines()[-1])
            line4 = int(file4.read().splitlines()[-1])

            out1 = open('out1.txt', 'w')
            out2 = open('out2.txt', 'w')
            out3 = open('out3.txt', 'w')
            out4 = open('out4.txt', 'w')

            totals = line1 + line2 + line3 + line4

            avgs = totals / total

            if(initial1):
                start = time.time()
                initial1 = 0
                initial2 = int(line2/avgs)
                if initial2 < mins:
                    initial2 = mins

            elif(initial2):
                initial2 = 0
                start = time.time()
                initial3 = int(line3/avgs)
                if initial3 < mins:
                    initial3 = mins

            elif(initial3):
                initial3 = 0
                start = time.time()
                initial4 = int(line4/avgs)
                if initial4 < mins:
                    initial4 = mins

            elif(initial4):
                initial4 = 0
                start = time.time()
                initial1 = int(line1/avgs)
                if initial1 < mins:
                    initial1 = mins


schedule_algo()
