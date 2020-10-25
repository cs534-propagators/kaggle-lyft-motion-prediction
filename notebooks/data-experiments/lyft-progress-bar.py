#>
import time
from datetime import datetime

#>
# helper to convert a timedelta to a string (dropping milliseconds)
def deltaToString(delta):
    timeObj = time.gmtime(delta.total_seconds())
    return time.strftime('%H:%M:%S', timeObj)

class ProgressBar:
    
    # constructor
    #   maxIterations: maximum number of iterations
    def __init__(self, maxIterations):
        self.maxIterations = maxIterations
        self.granularity = 100 # 1 whole percent
    
    # start the timer
    def start(self):
        self.start = datetime.now()
    
    # check the progress of the current iteration
    #   # currentIteration: the current iteration we are on
    def check(self, currentIteration, chunked=False):
        if currentIteration % round(self.maxIterations / self.granularity) == 0 or chunked:
            
            percentage = round(currentIteration / (self.maxIterations - self.maxIterations / self.granularity) * 100)
            
            current = datetime.now()
            
            # time calculations
            timeElapsed = (current - self.start)
            timePerStep = timeElapsed / (currentIteration + 1)
            totalEstimatedTime = timePerStep * self.maxIterations
            timeRemaining = totalEstimatedTime - timeElapsed
            
            # string formatting
            percentageStr = "{:>3}%  ".format(percentage)
            remainingStr = "Remaining: {}  ".format(deltaToString(timeRemaining))
            elapsedStr = "Elapsed: {}  ".format(deltaToString(timeElapsed))
            totalStr = "Total: {}\r".format(deltaToString(totalEstimatedTime))
            
            print(percentageStr + remainingStr + elapsedStr + totalStr, end="")

    def end(self):
        print()

#>
pb = ProgressBar(100)
pb.start()
pb.check(0)
pb.end()

#>
length = 1*10**7
pb = ProgressBar(length)
pb.start()
for i in range(0,length):
    pb.check(i)
pb.end()

#>
length = 80438
pb = ProgressBar(length)
pb.start()
for i in range(0,length):
    pb.check(i)
pb.end()

#>
length = 4853
pb = ProgressBar(length)
pb.start()
for i in range(0,length,10):
    pb.check(i)
pb.end()

#>
length = 4853
pb = ProgressBar(length)
pb.start()
for i in range(0,length,10):
    pb.check(i,True)
pb.end()

#>
length = 4853
pb = ProgressBar(length)
pb.start()
for i in range(0,length,10):
    for j in range(0, 10):
        pb.check(i+j)
pb.end()

#>
length = 4853
pb = ProgressBar(length)
pb.start()
for i in range(0,length,10):
    for j in range(i, i+10):
        pb.check(j)
pb.end()

#>

