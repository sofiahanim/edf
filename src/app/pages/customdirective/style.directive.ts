import { Directive, Renderer2 ,ElementRef,Input} from '@angular/core';

@Directive({
  selector: '[ngxStyle]'
})
export class StyleDirective {

  constructor(private element: ElementRef, private renderer: Renderer2) { }

  @Input() set ngxStyle(styles: Object){
    let entries = Object.entries(styles);

    for(let entry of entries){
      this.renderer.setStyle(this.element.nativeElement, entry[0], entry[1]);
    }
  }

}
