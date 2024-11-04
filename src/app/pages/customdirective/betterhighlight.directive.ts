import { Directive,ElementRef,HostBinding,HostListener,Input,Renderer2,OnInit } from '@angular/core';

@Directive({
  selector: '[ngxBetterhighlight]'
})
export class BetterhighlightDirective implements OnInit {

  constructor(private elemenet: ElementRef, private renderer: Renderer2) { }

  @Input()  defaultColor: string = 'transparent';
  @Input('ngxBetterhighlight')  highlightColor: string = 'pink';
  @Input() title: string = 'this is title';

  @HostBinding('style.backgroundColor') backgroundColor: string = this.defaultColor;
  @HostBinding('style.border') border: string = 'none';

  ngOnInit(){
    //this.renderer.setStyle(this.elemenet.nativeElement,'background-color','pink');
    //this.renderer.setStyle(this.elemenet.nativeElement,'border','1px solid red');
    this.backgroundColor = this.defaultColor;
    this.border = 'none';
  }

  @HostListener('mouseenter') mouseover(eventData: Event){
    this.backgroundColor = this.highlightColor;
    this.border = '1px solid red';
  }

  @HostListener('mouseleave') mouseleave(eventData: Event){
    this.backgroundColor = this.defaultColor;
    this.border = 'none';
  }


}
