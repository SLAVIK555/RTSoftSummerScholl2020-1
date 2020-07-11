SUMMARY = "Circle Buffer Demo"
SECTION = "examples"
LICENSE = "GPL"
APP_NAME = "cbdriver"
localdir = "/usr/local"
bindir = "${localdir}/bin"
TARGET_CC_ARCH += "${LDFLAGS}"
SRC_URI = "file://cbdriver.c \
file://Makefile \
"
S = "${WORKDIR}"
do_compile() {
make -f Makefile
}
do_install () {
install -m 0755 -d ${D}${localdir}
install -m 0755 -d ${D}${bindir}
cd ${S}
install -m 0755 ${APP_NAME} ${D}${bindir}
}
FILES_${PN}-dev = ""
FILES_${PN} = "${bindir}/*"
