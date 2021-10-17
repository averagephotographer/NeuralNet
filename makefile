JFLAGS = -d .
JC = javac
.SUFFIXES: .java .class
.java.class:
		$(JC) $(JFLAGS) $*.java

CLASSES = \
		background/Array.java \
		background/MiniBatch.java \
		Net.java

default: classes

classes: $(CLASSES:.java=.class)

clean:
		$(RM) *.class
