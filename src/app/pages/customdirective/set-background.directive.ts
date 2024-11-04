import { Directive, ElementRef, OnInit } from '@angular/core';  

@Directive({ selector: '[setBackground]' })

export class SetBackgroundDirective implements OnInit{

   // private el: ElementRef;
    constructor(private element: ElementRef) { 
        //this.el.nativeElement.style.backgroundColor = '#C8E6C9';
        this.element=element;
    }

    ngOnInit(){
        console.log('SetBackgroundDirective ngOnInit');
        this.element.nativeElement.style.backgroundColor = '#C8E6C9';
    }
}