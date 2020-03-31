cd .
mkdir snap

cd snap
git clone https://github.com/senbox-org/snap-engine.git
cd snap-engine
mvn clean  -DskipTests=true install

cd ..
git clone https://github.com/senbox-org/snap-desktop.git
cd snap-desktop
mvn clean  -DskipTests=true install

cd ..
git clone git clone https://github.com/senbox-org/s1tbx.git
cd s1tbx
mvn clean  -DskipTests=true install


