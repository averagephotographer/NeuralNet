JFLAGS = -d .
JC = javac
.SUFFIXES: .java .class
.java.class:
		$(JC) $(JFLAGS) $*.java

CLASSES = \
		background/A.java \
		background/MiniBatch.java \
		background/Network.java \
		Net.java \
		Main.java

default: classes

classes: $(CLASSES:.java=.class)

clean:
		$(RM) *.class
