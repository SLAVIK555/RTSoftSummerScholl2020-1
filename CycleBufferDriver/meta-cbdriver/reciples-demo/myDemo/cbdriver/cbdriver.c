#include <linux/kernel.h> /* Для printk() и т.д. */
#include <linux/module.h> /* Эта частичка древней магии, которая оживляет модули */
#include <linux/init.h> /* Определения макросов */
#include <linux/fs.h>
#include <asm/uaccess.h> /* put_user */
//#include <linux/ioctl.h>
//#include <sys/ioctl.h>  /* ioctl */


// Ниже мы задаём информацию о модуле, которую можно будет увидеть с помощью Modinfo
MODULE_LICENSE( "GPL" );
MODULE_SUPPORTED_DEVICE( "cbdriver" ); /* /dev/testdevice */

#define SUCCESS 0
#define DEVICE_NAME "cbdriver" /* Имя нашего устройства */
#define SIZE_BUF 256

// Поддерживаемые нашим устройством операции
static int device_open( struct inode *, struct file * );
static int device_release( struct inode *, struct file * );
static ssize_t device_read( struct file *, char *, size_t, loff_t * );
static ssize_t device_write( struct file *, const char *, size_t, loff_t * );

// Глобальные переменные, объявлены как static, воизбежание конфликтов имен.
static int major_number; /* Старший номер устройства нашего драйвера */
static int is_device_open = 0; /* Используется ли девайс ? */
//static char text[ 5 ] = "test\n"; /* Текст, который мы будет отдавать при обращении к нашему устройству */
//static char* text_ptr = text; /* Указатель на текущую позицию в тексте */

//кольцевой (циклический) буфер
unsigned char cycleBuf[SIZE_BUF];
unsigned char tail = 0;      //"указатель" хвоста буфера 
unsigned char head = 0;   //"указатель" головы буфера
unsigned char count = 0;  //счетчик символов

// Прописываем обработчики операций на устройством
static struct file_operations fops =
 {
  .read = device_read,
  .write = device_write,
  .open = device_open,
  .release = device_release
  //.ioctl = device_ioctl
 };

//"очищаем" буфер
void FlushBuf(void)
{
  tail = 0;
  head = 0;
  count = 0;
}

//положить символ в буфер
void PutChar(unsigned char sym)
{
  if (count < SIZE_BUF){   //если в буфере еще есть место
      cycleBuf[tail] = sym;    //помещаем в него символ
      count++;                    //инкрементируем счетчик символов
      tail++;                           //и индекс хвоста буфера
      if (tail == SIZE_BUF) tail = 0;
    }
}

//взять символ из буфера
unsigned char GetChar(void)
{
   unsigned char sym = 0;
   if (count > 0){                            //если буфер не пустой
      sym = cycleBuf[head];              //считываем символ из буфера
      count--;                                   //уменьшаем счетчик символов
      head++;                                  //инкрементируем индекс головы буфера
      if (head == SIZE_BUF) head = 0;
   }
   return sym;
}

// Функция загрузки модуля. Входная точка. Можем считать что это наш main()
static int __init test_init( void )
{
 printk( KERN_ALERT "TEST driver loaded!\n" );

 // Регистрируем устройсво и получаем старший номер устройства
 major_number = register_chrdev( 0, DEVICE_NAME, &fops );

 if ( major_number < 0 )
 {
  printk( "Registering the character device failed with %d\n", major_number );
  return major_number;
 }

 // Сообщаем присвоенный нам старший номер устройства
 printk( "Test module is loaded!\n" );

 printk( "Please, create a dev file with 'mknod /dev/test c %d 0'.\n", major_number );

 return SUCCESS;
}

// Функция выгрузки модуля
static void __exit test_exit( void )
{
 // Освобождаем устройство
 unregister_chrdev( major_number, DEVICE_NAME );

 printk( KERN_ALERT "Test module is unloaded!\n" );
}

// Указываем наши функции загрузки и выгрузки
module_init( test_init );
module_exit( test_exit );

static int device_open( struct inode *inode, struct file *file )
{
 //text_ptr = text;

 if ( is_device_open )
  return -EBUSY;

 is_device_open++;

 return SUCCESS;
}

static int device_release( struct inode *inode, struct file *file )
{
 is_device_open--;
 return SUCCESS;
}

static ssize_t

device_write( struct file *filp, const char *buff, size_t len, loff_t * off )
{
  int byte_write = 0;

  //if ( *text_ptr == 0 )//text_ptr = text[], тот массив. из которого мы читаем
  //  return 0;

  while ( len  )
  {
    PutChar(buff[byte_write]);
    //i++;
    //put_user( *( text_ptr++ ), buffer++ );
    len--;
    byte_write++;
  }

  return byte_write;
}

static ssize_t device_read( struct file *filp, char *buffer, size_t length, loff_t * offset ){
  int byte_read = 0;

  //if ( *text_ptr == 0 )//text_ptr = text[], тот массив. из которого мы читаем
  //  return 0;

  while ( length  )
  {
    GetChar();
    //put_user( *( text_ptr++ ), buffer++ );
    length--;
    byte_read++;
  }

  return byte_read;
}